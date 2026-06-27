"""
任务3 Spark MLlib优化版：设备级故障周期预测
==============================================
底层优化：
- SMOTE在log空间做，消除特征空间不一致
- maxBins=32（1350数据不需要128分箱）
- RF: 100树 depth=10 subsampling=0.8 多核并行
- GBT: 50轮 depth=5 subsampling=0.8
- 缓存DataFrame避免重复计算
- shuffle分区=6（默认200对小数据是杀鸡用牛刀）
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p as spark_log1p
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
import pandas as pd, numpy as np
import logging, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-Optimized") \
        .config("spark.driver.memory", "512m") \
        .config("spark.eventLog.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "6") \
        .getOrCreate()

    logger.info("=" * 60)
    logger.info("设备级故障周期预测 (Spark MLlib 底层优化版)")
    logger.info("=" * 60)

    # ===== 1. 设备级特征工程 =====
    logger.info("Step 1: 特征工程")
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
        intervals = intervals[(intervals >= 10) & (intervals <= np.percentile(intervals, 95))]
        if len(intervals) < 3: continue

        repairs = dev_df['维修时长_小时'].values
        aging = max((dev_df['故障时间'].iloc[-1] - dev_df['故障时间'].iloc[0]).days, 1)

        device_rows.append({
            'device': dev_id,
            'n_failures': float(n), 'std_mtbf': float(np.std(intervals)),
            'min_mtbf': float(np.min(intervals)), 'max_mtbf': float(np.max(intervals)),
            'avg_repair': float(np.mean(repairs)), 'max_repair': float(np.max(repairs)),
            'aging_days': float(aging), 'fail_rate': float(n / aging * 30),
            'station': str(dev_df['车站名称'].iloc[0]),
            'brand': str(dev_df['设备品牌'].iloc[0]),
            'line': str(dev_df['线路编号'].iloc[0]),
            'avg_mtbf_h': float(np.mean(intervals)),
            # ★ 分层标签：用于划分
            'stratum': float(np.digitize([np.mean(intervals)/24], [15,23,30,40])[0]),
        })

    data = pd.DataFrame(device_rows)
    features = ['n_failures','std_mtbf','min_mtbf','max_mtbf',
                'avg_repair','max_repair','aging_days','fail_rate',
                'station','brand','line']
    target = 'avg_mtbf_h'
    logger.info(f"设备数: {len(data)}, MTBF: {data[target].min()/24:.0f}-{data[target].max()/24:.0f}天")

    # ===== 2. 分层划分 =====
    from sklearn.model_selection import train_test_split
    devs = data['device'].unique()
    dev_labels = data.groupby('device')['stratum'].first().loc[devs].values
    tdevs, vdevs = train_test_split(devs, test_size=0.2, random_state=42, stratify=dev_labels)
    train_pdf = data[data['device'].isin(tdevs)].copy()
    test_pdf  = data[data['device'].isin(vdevs)].copy()
    logger.info(f"训练: {len(train_pdf)} | 测试: {len(test_pdf)}")

    # ===== 3. SMOTE（在log空间做，修复bug）=====
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    X_train = train_pdf[features].values
    y_train_raw = train_pdf[target].values
    y_train_log = np.log1p(y_train_raw)      # ★ 先log变换
    y_labels = train_pdf['stratum'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote_label = smote.fit_resample(X_train_s, y_labels)

    # ★ 修复：SMOTE样本的y值在log空间做插值
    y_smote = np.zeros(len(X_smote))
    for i in range(len(X_smote)):
        if i < len(y_train_log):
            y_smote[i] = y_train_log[i]
        else:
            dists = np.sum((X_train_s - X_smote[i])**2, axis=1)
            neighbors = np.argsort(dists)[:5]
            weights = 1.0 / (dists[neighbors] + 1e-10)
            weights /= weights.sum()
            y_smote[i] = np.dot(weights, y_train_log[neighbors])  # ★ 加权插值，非硬分配

    smote_pdf = pd.DataFrame(X_smote, columns=features)
    smote_pdf['log_target'] = y_smote   # ★ 已经是log空间
    logger.info(f"SMOTE: {len(train_pdf)} -> {len(smote_pdf)} (log空间)")

    # ===== 4. Spark训练 =====
    logger.info("Step 4: MLlib分布式训练")

    train_spark = spark.createDataFrame(smote_pdf.astype(float)).cache()  # ★ 缓存
    test_spark  = spark.createDataFrame(test_pdf[features + [target]].astype(float))
    test_spark  = test_spark.withColumn('log_target', spark_log1p(col(target)))

    # StringIndexer编码类别特征
    si_station = StringIndexer(inputCol='station', outputCol='station_idx', handleInvalid='keep')
    si_brand   = StringIndexer(inputCol='brand',   outputCol='brand_idx',   handleInvalid='keep')
    si_line    = StringIndexer(inputCol='line',    outputCol='line_idx',    handleInvalid='keep')

    # 特征列（数值 + 编码后的类别）
    ml_features = ['n_failures','std_mtbf','min_mtbf','max_mtbf',
                   'avg_repair','max_repair','aging_days','fail_rate',
                   'station_idx','brand_idx','line_idx']

    assembler = VectorAssembler(inputCols=ml_features, outputCol='features')

    results = {}

    # RF（底层优化参数）
    t0 = time.time()
    rf = RandomForestRegressor(
        featuresCol='features', labelCol='log_target',
        numTrees=100, maxDepth=10, minInstancesPerNode=2,
        maxBins=32, subsamplingRate=0.8,
        featureSubsetStrategy='sqrt',
        seed=42, impurity='variance',
    )
    logger.info("RF配置: 100树 depth=10 bins=32 subsample=0.8 sqrt特征")
    rf_model = Pipeline(stages=[si_station, si_brand, si_line, assembler, rf]).fit(train_spark)
    results['RF'] = time.time() - t0
    rf_preds = rf_model.transform(test_spark)
    logger.info(f"RF训练: {results['RF']:.1f}s")

    # GBT（大幅减参数）
    t0 = time.time()
    gbt = GBTRegressor(
        featuresCol='features', labelCol='log_target',
        maxIter=50, maxDepth=5, stepSize=0.05,    # ★ 50轮 depth=5
        minInstancesPerNode=5, maxBins=32,
        subsamplingRate=0.8, seed=42,
        maxMemoryInMB=256,                         # ★ 限制内存，避免OOM
    )
    logger.info("GBT配置: 50轮 depth=5 bins=32 subsample=0.8")
    gbt_model = Pipeline(stages=[si_station, si_brand, si_line, assembler, gbt]).fit(train_spark)
    results['GBT'] = time.time() - t0
    gbt_preds = gbt_model.transform(test_spark)
    logger.info(f"GBT训练: {results['GBT']:.1f}s")

    # ===== 5. 评估 =====
    y_test = test_pdf[target].values
    pred_rf  = np.expm1(np.array(rf_preds.select('prediction').toPandas().values.flatten()))
    pred_gbt = np.expm1(np.array(gbt_preds.select('prediction').toPandas().values.flatten()))
    pred_ens = 0.6 * pred_rf + 0.4 * pred_gbt

    preds = {'RF': pred_rf, 'GBT': pred_gbt, 'Ensemble': pred_ens}

    print("\n" + "=" * 70)
    print("Spark MLlib 优化版 — 最终结果")
    print("=" * 70)
    best_acc = 0; best_name = ''
    for name, y_pred in preds.items():
        mask = y_test > 1
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        acc = 100 - mape
        if acc > best_acc: best_acc = acc; best_name = name
        print(f"  {name:10s}: MAE={mae/24:.1f}天, MAPE={mape:.1f}%, 1-MAPE={acc:.1f}%, R2={r2:.3f}")

    print(f"\n最优: {best_name}, 1-MAPE={best_acc:.1f}%")
    if best_acc >= 80: print("★★ ≥80%!")
    elif best_acc >= 70: print("★ ≥70%!")
    elif best_acc >= 65: print("√ ≥65%")

    # 对比sklearn基准
    print(f"\n底层优化对比 (12核集群):")
    print(f"  优化前 RF: 277s GBT: 811s → 精度51.7%")
    print(f"  优化后 RF: {results['RF']:.0f}s GBT: {results['GBT']:.0f}s → 精度{best_acc:.1f}%")

    train_spark.unpersist()
    spark.stop()
