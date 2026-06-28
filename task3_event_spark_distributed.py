"""
事件级回归 — 分布式numpy RF (Spark版)
=====================================
真正分布式: 广播数据到Worker, 每棵树独立Bootstrap+训练, Driver收集集成

对比单机版:
  单机: for i in range(150): train_tree()          ← 串行, 只用1核
  分布式: rdd.map(train_tree).collect()             ← 3Worker×4核并行

关键设计:
  - 广播变量传数据 (X/y只发一次, task只带元数据)
  - 闭包捕获广播变量 (PySpark cloudpickle序列化)
  - 两阶段: scout RF(50树)→特征重要性→主RF(150树)
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42

# ========== numpy决策树 (模块级定义, spark-submit发送到Worker) ==========
class NumPyTree:
    """精确分裂决策树, 算法等价sklearn CART"""
    def __init__(self, max_depth=15, min_samples=3):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.importances_ = np.zeros(self.n_features_)
        self.tree_ = self._build(X, y, 0)
        return self

    def _best_split(self, X, y):
        n = len(y)
        best_gain = 1e-12
        best = (None, None, 0.0)
        n_feats = max(1, int(np.sqrt(X.shape[1])))

        for f in np.random.choice(X.shape[1], n_feats, replace=False):
            x = X[:, f]
            idx = np.argsort(x)
            xs, ys = x[idx], y[idx]
            cs = np.cumsum(ys)
            total_var = np.var(ys) * n

            for i in range(self.min_samples, n - self.min_samples):
                if xs[i] == xs[i + 1]:
                    continue
                nl, nr = i + 1, n - i - 1
                sl, sr = cs[i], cs[-1] - cs[i]
                mse = np.sum((ys[:i+1] - sl/nl)**2) + np.sum((ys[i+1:] - sr/nr)**2)
                gain = total_var - mse
                if gain > best_gain:
                    best_gain = gain
                    best = (f, (xs[i] + xs[i+1]) / 2.0, gain)
        return best  # (feature, threshold, gain)

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples * 2:
            return np.mean(y)
        f, t, gain = self._best_split(X, y)
        if f is None:
            return np.mean(y)
        self.importances_[f] += gain
        left = X[:, f] <= t
        right = ~left
        if left.sum() < self.min_samples or right.sum() < self.min_samples:
            return np.mean(y)
        return {'f': f, 't': t,
                'L': self._build(X[left], y[left], depth + 1),
                'R': self._build(X[right], y[right], depth + 1)}

    def predict(self, X):
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self.tree_
            while isinstance(node, dict):
                node = node['L'] if x[node['f']] <= node['t'] else node['R']
            out[i] = node
        return out


# ========== 主流程 ==========
if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-Event-Distributed-RF") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "512m") \
        .config("spark.eventLog.enabled", "false") \
        .getOrCreate()

    sc = spark.sparkContext
    logger.info("=" * 60)
    logger.info(f"事件级故障间隔预测 — 分布式numpy RF (Spark)")
    logger.info(f"集群: Master + 3Worker, 每节点4核768MB")
    logger.info("=" * 60)

    # ===== 1. 数据加载 + 事件级特征工程 (Driver端) =====
    logger.info("Step 1: 事件级特征工程 (Driver)")
    pdf = pd.read_csv('/root/cleaned_afc_data.csv', encoding='utf-8-sig')
    for c in ['故障时间', '维修开始时间', '维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')
    pdf = pdf.sort_values(['设备编号', '故障时间'])

    from sklearn.preprocessing import LabelEncoder
    all_stations = pdf['车站名称'].astype(str)
    all_brands = pdf['设备品牌'].astype(str)
    le_s = LabelEncoder().fit(all_stations)
    le_b = LabelEncoder().fit(all_brands)

    rows = []
    for dev_id, dev_df in pdf.groupby('设备编号'):
        dev_df = dev_df.sort_values('故障时间').reset_index(drop=True)
        intervals = dev_df['故障时间'].diff().dt.total_seconds() / 3600
        hist = []
        for i in range(1, len(dev_df)):
            tgt = intervals.iloc[i]
            if np.isnan(tgt) or tgt <= 0:
                continue
            cur = dev_df.iloc[i]
            nh = len(hist)
            aging = max((cur['故障时间'] - dev_df['故障时间'].iloc[0]).total_seconds() / 86400, 1)
            rows.append({
                'device': dev_id,
                'hour': cur['故障时间'].hour,
                'weekday': cur['故障时间'].dayofweek,
                'month': cur['故障时间'].month,
                'repair_dur': cur['维修时长_小时'],
                'response': max((cur['维修开始时间'] - cur['故障时间']).total_seconds() / 3600, 0),
                'rtype': 1 if cur['维修类型'] == 'CBM' else 0,
                'n_hist': nh,
                'last_int': hist[-1] if hist else 0,
                'avg_int': np.mean(hist) if hist else 0,
                'aging_days': aging,
                'fail_rate': nh / aging,
                'station_enc': le_s.transform([str(cur['车站名称'])])[0],
                'brand_enc': le_b.transform([str(cur['设备品牌'])])[0],
                'target_h': tgt,
            })
            hist.append(tgt)

    data = pd.DataFrame(rows)
    data = data[data['last_int'] > 0]
    data = data[(data['target_h'] >= 10) & (data['target_h'] <= data['target_h'].quantile(0.99))]
    logger.info(f"  事件数: {len(data)}")

    feats = ['hour', 'weekday', 'month', 'repair_dur', 'response', 'rtype',
             'n_hist', 'last_int', 'avg_int', 'aging_days', 'fail_rate',
             'station_enc', 'brand_enc']

    # ===== 2. 划分 + 标准化 (Driver端) =====
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    edevs = data['device'].unique()
    etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=RANDOM_SEED)
    etr = data[data['device'].isin(etdevs)]
    ete = data[data['device'].isin(evdevs)]

    X_tr = etr[feats].values.astype(np.float64)
    y_tr = etr['target_h'].values.astype(np.float64)
    X_te = ete[feats].values.astype(np.float64)
    y_te = ete['target_h'].values.astype(np.float64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)
    logger.info(f"  训练: {len(etr)} | 测试: {len(ete)} | 特征: {len(feats)}")

    # ===== 3. Stage 1: 分布式scout RF → 特征重要性 =====
    logger.info("=" * 60)
    logger.info("Stage 1: 分布式scout RF (50树×depth=8) → 特征重要性")
    logger.info("=" * 60)

    # ★ 广播数据: 每个Worker只收一次, task不带数据
    X_bc = sc.broadcast(X_tr_s)
    y_bc = sc.broadcast(y_tr_log)
    logger.info(f"  广播 {X_tr_s.nbytes/1024**2:.1f}MB 特征矩阵到所有Worker")

    SCOUT_TREES = 50
    SCOUT_DEPTH = 8
    SCOUT_MIN = 5
    scout_parts = min(SCOUT_TREES, 12)

    # ★ 闭包捕获广播变量: PySpark cloudpickle序列化, Worker通过ID查找数据
    def _train_scout(task):
        tree_idx, md, ms, seed = task
        X = X_bc.value   # ← 广播变量引用, 不是数据拷贝
        y = y_bc.value
        np.random.seed(seed)
        n = len(y)
        idx = np.random.choice(n, n, replace=True)
        tree = NumPyTree(md, ms)
        tree.fit(X[idx], y[idx])
        return tree

    scout_tasks = [(i, SCOUT_DEPTH, SCOUT_MIN, RANDOM_SEED + i)
                   for i in range(SCOUT_TREES)]

    t0 = time.time()
    scout_trees = sc.parallelize(scout_tasks, numSlices=scout_parts) \
                     .map(_train_scout) \
                     .collect()
    t_scout = time.time() - t0

    # 特征重要性 (Driver端, 毫秒级)
    importances = np.mean([t.importances_ for t in scout_trees], axis=0)
    importances = importances / importances.sum()
    ranked = np.argsort(importances)[::-1]
    top10_idx = ranked[:10]
    top10_names = [feats[i] for i in top10_idx]

    logger.info(f"  Scout完成: {t_scout:.0f}秒 ({SCOUT_TREES}棵树/{scout_parts}分区)")
    logger.info(f"  特征重要性 Top10:")
    for rank, idx in enumerate(ranked):
        marker = " ← Top10" if idx in top10_idx else ""
        logger.info(f"    {rank+1:2d}. {feats[idx]:12s} = {importances[idx]:.4f}{marker}")

    # 清理scout广播变量,释放Worker内存
    X_bc.destroy(); y_bc.destroy()

    # ===== 4. Stage 2: 分布式主RF (Top10特征) =====
    logger.info("=" * 60)
    logger.info("Stage 2: 分布式主RF (150树×depth=15) → Top10特征")
    logger.info("=" * 60)

    X_tr_top = X_tr_s[:, top10_idx]
    X_te_top = X_te_s[:, top10_idx]

    # 重新广播Top10数据
    X_bc2 = sc.broadcast(X_tr_top)
    y_bc2 = sc.broadcast(y_tr_log)
    logger.info(f"  广播 {X_tr_top.nbytes/1024**2:.1f}MB (Top10特征) 到所有Worker")

    MAIN_TREES = 150
    MAIN_DEPTH = 15
    MAIN_MIN = 3
    main_parts = min(36, MAIN_TREES)

    def _train_main(task):
        tree_idx, md, ms, seed = task
        X = X_bc2.value  # ← Top10特征的广播变量
        y = y_bc2.value
        np.random.seed(seed)
        n = len(y)
        idx = np.random.choice(n, n, replace=True)
        tree = NumPyTree(md, ms)
        tree.fit(X[idx], y[idx])
        return tree

    main_tasks = [(i, MAIN_DEPTH, MAIN_MIN, RANDOM_SEED + 1000 + i)
                  for i in range(MAIN_TREES)]

    t0 = time.time()
    main_trees = sc.parallelize(main_tasks, numSlices=main_parts) \
                   .map(_train_main) \
                   .collect()
    t_main = time.time() - t0

    # 清理
    X_bc2.destroy(); y_bc2.destroy()

    logger.info(f"  主RF完成: {t_main:.0f}秒 ({MAIN_TREES}棵树/{main_parts}分区)")
    logger.info(f"  估计加速比: ~{main_parts/3:.0f}x (vs单机串行)")

    # ===== 5. 集成预测 (Driver端) =====
    logger.info("Step 5: 集成预测 (Driver)")

    def rf_predict(trees, X):
        preds = np.column_stack([t.predict(X) for t in trees])
        return preds.mean(axis=1)

    y_pred_log = rf_predict(main_trees, X_te_top)
    y_pred = np.expm1(y_pred_log)

    # ===== 6. 保存模型 =====
    with open('/root/event_numpy_rf.pkl', 'wb') as f:
        pickle.dump({
            'trees': main_trees,
            'features': top10_names,
            'scaler_mean': scaler.mean_[top10_idx].tolist(),
            'scaler_scale': scaler.scale_[top10_idx].tolist(),
            'target_transform': 'log1p',
            'train_method': 'spark_distributed',
            'n_trees': MAIN_TREES,
            'max_depth': MAIN_DEPTH,
            'min_samples': MAIN_MIN,
            'scout_trees': SCOUT_TREES,
            'importances': importances.tolist(),
            'all_feature_names': feats,
        }, f)
    logger.info("模型已保存: /root/event_numpy_rf.pkl")

    # ===== 7. 评估 =====
    mask = y_te > 1
    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    mape = np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100
    r2 = 1 - np.sum((y_te - y_pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    acc = 100 - mape

    print("\n" + "=" * 70)
    print("分布式numpy RF (Spark) — 事件级故障间隔预测")
    print("=" * 70)
    print(f"  集群: 3Worker × 4核 = 12核, 每节点768MB")
    print(f"  Scout: {SCOUT_TREES}树×depth={SCOUT_DEPTH} → 选Top10特征")
    print(f"  主RF: {MAIN_TREES}树×depth={MAIN_DEPTH}×ms={MAIN_MIN} → {len(top10_names)}特征")
    print(f"  Scout时间: {t_scout:.0f}秒 | 主RF时间: {t_main:.0f}秒 | 总计: {t_scout + t_main:.0f}秒")
    print(f"  ───────────────────────────────────────")
    print(f"  MAE:  {mae:.1f}h ({mae/24:.1f}天)")
    print(f"  RMSE: {rmse:.1f}h ({rmse/24:.1f}天)")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  1-MAPE: {acc:.1f}%")
    print(f"  R²:   {r2:.3f}")
    print(f"  ───────────────────────────────────────")
    if acc >= 80:
        print(f"  ★★ ≥80% 额外加分!")
    elif acc >= 70:
        print(f"  ★ ≥70% 加分!")
    elif acc >= 65:
        print(f"  √ ≥65% 达标")
    else:
        print(f"  ✗ <65%")

    spark.stop()
    print("\n完成!")
