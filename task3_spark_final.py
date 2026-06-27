"""
任务3 Spark MLlib版：设备级故障周期预测（集群分布式训练）
============================================================
- 设备级特征（pandas预处理）→ Spark MLlib分布式训练
- RF + GBT + LinearRegression 对比
- log1p目标变换 + 分层划分 + SMOTE
- 运行: spark-submit task3_spark_final.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p as spark_log1p
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml import Pipeline
import pandas as pd, numpy as np
import logging, time, pickle as pkl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-DeviceRegression") \
        .config("spark.driver.memory", "1g") \
        .config("spark.eventLog.enabled", "false") \
        .getOrCreate()

    logger.info("=" * 60)
    logger.info("设备级故障周期预测 (Spark MLlib 分布式)")
    logger.info("=" * 60)

    # ===== 1. 设备级特征工程 (Driver pandas) =====
    logger.info("Step 1: 设备级特征工程")
    pdf = pd.read_csv('/root/cleaned_afc_data.csv', encoding='utf-8-sig')
    for c in ['故障时间', '维修开始时间', '维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')

    device_rows = []
    for dev_id, dev_df in pdf.groupby('设备编号'):
        dev_df = dev_df.sort_values('故障时间'); n = len(dev_df)
        if n < 5: continue
        intervals = []
        for i in range(1, n):
            delta = (dev_df['故障时间'].iloc[i] - dev_df['故障时间'].iloc[i-1]).total_seconds() / 3600
            if delta > 0: intervals.append(delta)
        if len(intervals) < 3: continue
        intervals = np.array(intervals)
        mask = (intervals >= 10) & (intervals <= np.percentile(intervals, 95))
        intervals = intervals[mask]
        if len(intervals) < 3: continue

        repairs = dev_df['维修时长_小时'].values
        aging = max((dev_df['故障时间'].iloc[-1] - dev_df['故障时间'].iloc[0]).days, 1)

        device_rows.append({
            'device': dev_id, 'n_failures': float(n),
            'std_mtbf_h': float(np.std(intervals)),
            'min_mtbf_h': float(np.min(intervals)),
            'max_mtbf_h': float(np.max(intervals)),
            'avg_repair_h': float(np.mean(repairs)),
            'max_repair_h': float(np.max(repairs)),
            'aging_days': float(aging),
            'fail_rate_monthly': float(n / aging * 30),
            'station_enc': float(hash(str(dev_df['车站名称'].iloc[0])) % 1000),
            'brand_enc': float(hash(str(dev_df['设备品牌'].iloc[0])) % 1000),
            'line_enc': float(hash(str(dev_df['线路编号'].iloc[0])) % 1000),
            'avg_mtbf_h': float(np.mean(intervals)),
            'label_stratify': float(pd.cut([np.mean(intervals)/24], bins=[0,15,23,30,40,999], labels=False)[0]),
        })

    data = pd.DataFrame(device_rows)
    target_col = 'avg_mtbf_h'
    features = ['n_failures', 'std_mtbf_h', 'min_mtbf_h', 'max_mtbf_h',
                'avg_repair_h', 'max_repair_h', 'aging_days', 'fail_rate_monthly',
                'station_enc', 'brand_enc', 'line_enc']
    logger.info(f"设备数: {len(data)}, MTBF: {data[target_col].min()/24:.0f}-{data[target_col].max()/24:.0f}天")

    # ===== 2. 划分 (Driver) =====
    logger.info("Step 2: 分层划分")
    from sklearn.model_selection import train_test_split
    devs = data['device'].unique()
    dev_labels = data.groupby('device')['label_stratify'].first().loc[devs].values
    tdevs, vdevs = train_test_split(devs, test_size=0.2, random_state=42, stratify=dev_labels)
    train_pdf = data[data['device'].isin(tdevs)].copy()
    test_pdf  = data[data['device'].isin(vdevs)].copy()
    logger.info(f"训练: {len(train_pdf)} | 测试: {len(test_pdf)}")

    # ===== 3. SMOTE (Driver) =====
    logger.info("Step 3: SMOTE均衡")
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    X_tr_raw = train_pdf[features].values
    y_tr_raw = train_pdf[target_col].values
    y_tr_labels = train_pdf['label_stratify'].values

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_scaled, y_tr_labels)

    # 取SMOTE生成样本对应的原始y值
    smote_y = np.zeros(len(X_tr_smote))
    for i in range(len(X_tr_smote)):
        if i < len(y_tr_raw):
            smote_y[i] = y_tr_raw[i]
        else:
            dists = np.sum((X_tr_scaled - X_tr_smote[i])**2, axis=1)
            smote_y[i] = y_tr_raw[np.argmin(dists)]

    smote_pdf = pd.DataFrame(X_tr_smote, columns=features)
    smote_pdf[target_col] = smote_y
    logger.info(f"SMOTE: {len(train_pdf)} -> {len(smote_pdf)}")

    # ===== 4. Spark MLlib训练 =====
    logger.info("Step 4: Spark MLlib分布式训练")

    train_df = spark.createDataFrame(smote_pdf.astype(float))
    test_df  = spark.createDataFrame(test_pdf[features + [target_col]].astype(float))

    # log1p变换目标
    train_df = train_df.withColumn('log_target', spark_log1p(col(target_col)))
    test_df  = test_df.withColumn('log_target', spark_log1p(col(target_col)))

    # 特征向量（RF/GBT不需要标准化，直接拼接原始特征）
    assembler = VectorAssembler(inputCols=features, outputCol='features')

    results = {}

    # RF
    logger.info("--- RandomForest ---")
    t0 = time.time()
    rf = RandomForestRegressor(featuresCol='features', labelCol='log_target',
                                numTrees=200, maxDepth=15, minInstancesPerNode=3,
                                maxBins=128, seed=42)
    rf_model = Pipeline(stages=[assembler, rf]).fit(train_df)
    rf_preds = rf_model.transform(test_df)
    results['RF'] = time.time() - t0
    logger.info(f"  RF: {results['RF']:.1f}s")

    # GBT
    logger.info("--- GBT ---")
    t0 = time.time()
    gbt = GBTRegressor(featuresCol='features', labelCol='log_target',
                        maxIter=200, maxDepth=8, stepSize=0.05,
                        minInstancesPerNode=3, maxBins=128, seed=42)
    gbt_model = Pipeline(stages=[assembler, gbt]).fit(train_df)
    gbt_preds = gbt_model.transform(test_df)
    results['GBT'] = time.time() - t0
    logger.info(f"  GBT: {results['GBT']:.1f}s")

    # LinearRegression (基线)
    logger.info("--- LinearRegression ---")
    t0 = time.time()
    lr = LinearRegression(featuresCol='features', labelCol='log_target', maxIter=50)
    lr_model = Pipeline(stages=[assembler, lr]).fit(train_df)
    lr_preds = lr_model.transform(test_df)
    results['LinearReg'] = time.time() - t0
    logger.info(f"  LinearReg: {results['LinearReg']:.1f}s")

    # ===== 5. 评估 =====
    logger.info("Step 5: 评估")

    y_test = test_pdf[target_col].values
    preds = {
        'RF': np.expm1(np.array(rf_preds.select('prediction').toPandas().values.flatten())),
        'GBT': np.expm1(np.array(gbt_preds.select('prediction').toPandas().values.flatten())),
        'LinearReg': np.expm1(np.array(lr_preds.select('prediction').toPandas().values.flatten())),
    }
    y_ens = 0.6 * preds['RF'] + 0.4 * preds['GBT']
    preds['Ensemble'] = y_ens

    print("\n" + "=" * 70)
    print("Spark MLlib 分布式训练 — 设备级故障周期预测")
    print("=" * 70)
    best_acc = 0; best_name = ''
    for name in ['RF', 'GBT', 'Ensemble', 'LinearReg']:
        y_pred = preds[name]
        mask = y_test > 1
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        acc = 100 - mape
        marker = ' ★' if acc > best_acc else ''
        if acc > best_acc: best_acc = acc; best_name = name
        print(f"  {name:12s}: MAE={mae/24:.1f}天, MAPE={mape:.1f}%, 1-MAPE={acc:.1f}%, R2={r2:.3f}{marker}")

    print(f"\n最优模型: {best_name}, 1-MAPE = {best_acc:.1f}%")
    if best_acc >= 80: print("★★ 精度≥80% 额外加分!")
    elif best_acc >= 70: print("★ 精度≥70% 加分!")
    elif best_acc >= 65: print("√ 精度≥65% 达标")

    # 保存
    rf_model.write().overwrite().save('/root/model_rf_mlib_final')
    logger.info("模型已保存: /root/model_rf_mlib_final")

    spark.stop()
    print("完成!")
