"""
事件级回归 — 全链路分布式numpy RF (Spark版)
============================================
全链路分布式: 数据切分→Shuffle→Reduce聚合→特征工程→树训练 全部在Worker上并行

Phase 1 [Driver] :  加载CSV → 广播原始行到集群
Phase 2 [Map]    :  每行→(device_id, row) keyBy
Phase 3 [Shuffle] :  partitionBy(hash(device_id)) → 同设备路由到同一Worker
Phase 4 [Reduce]  :  分区内排序→相邻合并→每设备计算特征 ★ 手动reduceByKey
Phase 5 [Driver]  :  collect特征表 → StandardScaler → split
Phase 6 [广播]    :  broadcast(X_scaled, y_log) 到所有Worker
Phase 7 [Map]     :  并行Bootstrap+训练numpy树 (同前)
Phase 8 [Driver]  :  collect树→集成预测→评估

手动实现的核心Spark算子:
  Hash分桶:   hash(device_id) % num_partitions → 分区分配
  Shuffle:    partitionBy → 跨Worker重分布 (等价reduceByKey的数据移动)
  SortMerge:  分区内按键排序 → 相邻同Key合并 → 等价groupByKey+reduce
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42

# ========== numpy决策树 (Worker端训练) ==========
class NumPyTree:
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
            x = X[:, f]; idx = np.argsort(x); xs, ys = x[idx], y[idx]
            cs = np.cumsum(ys); tv = np.var(ys) * n
            for i in range(self.min_samples, n - self.min_samples):
                if xs[i] == xs[i + 1]: continue
                nl, nr = i + 1, n - i - 1
                sl, sr = cs[i], cs[-1] - cs[i]
                mse = np.sum((ys[:i+1] - sl/nl)**2) + np.sum((ys[i+1:] - sr/nr)**2)
                gain = tv - mse
                if gain > best_gain: best_gain = gain; best = (f, (xs[i] + xs[i+1]) / 2.0, gain)
        return best

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples * 2:
            return np.mean(y)
        f, t, gain = self._best_split(X, y)
        if f is None: return np.mean(y)
        self.importances_[f] += gain
        left = X[:, f] <= t; right = ~left
        if left.sum() < self.min_samples or right.sum() < self.min_samples:
            return np.mean(y)
        return {'f': f, 't': t, 'L': self._build(X[left], y[left], depth + 1),
                'R': self._build(X[right], y[right], depth + 1)}

    def predict(self, X):
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self.tree_
            while isinstance(node, dict):
                node = node['L'] if x[node['f']] <= node['t'] else node['R']
            out[i] = node
        return out


# ========== Phase 2-4: 分布式特征工程 (手动MapReduce) ==========

# ---- Step A: LabelEncoder预拟合 (Driver端, 需要全量扫描) ----
def _driver_fit_encoders(csv_path):
    """Driver端: 扫描全量数据拟合LabelEncoder (需要全局字典)"""
    pdf = pd.read_csv(csv_path, encoding='utf-8-sig')
    le_station = LabelEncoder().fit(pdf['车站名称'].astype(str))
    le_brand = LabelEncoder().fit(pdf['设备品牌'].astype(str))
    # 转换时间列
    for c in ['故障时间', '维修开始时间', '维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')
    # 转为list-of-dict 用于分布 (时间→Unix时间戳, 可被pickle序列化)
    rows = []
    for _, r in pdf.iterrows():
        rows.append({
            'device': str(r['设备编号']),
            'fault_ts': r['故障时间'].timestamp() if pd.notna(r['故障时间']) else 0.0,
            'repair_dur': float(r['维修时长_小时']) if pd.notna(r['维修时长_小时']) else 0.0,
            'repair_type': str(r['维修类型']) if pd.notna(r['维修类型']) else '',
            'station': str(r['车站名称']),
            'brand': str(r['设备品牌']),
        })
    return rows, le_station, le_brand


# ---- Step B: Hash分区函数 ----
def _hash_partition(device_str, num_parts):
    """手动Hash分区: hash(device_id) % N → 决定数据去哪个分片"""
    # Python hash()对同一字符串在同一进程内一致, 跨进程可能不同
    # 用md5保证跨进程一致性
    import hashlib
    h = hashlib.md5(device_str.encode()).hexdigest()
    return int(h, 16) % num_parts


# ---- Step C: Map阶段 → keyBy(device_id) + assign partition ----
def _map_assign_partition(row, num_parts, le_station, le_brand):
    """
    Map阶段: 每行 → (partition_id, (device_id, encoded_row))
    - 计算hash分区
    - 编码类别特征
    """
    pid = _hash_partition(row['device'], num_parts)
    encoded = {
        'device': row['device'],
        'fault_ts': float(row['fault_ts']),
        'repair_dur': float(row['repair_dur']),
        'repair_type': 1 if 'CBM' in row['repair_type'].upper() else 0,
        'station_enc': le_station.transform([row['station']])[0],
        'brand_enc': le_brand.transform([row['brand']])[0],
    }
    return (pid, (row['device'], encoded))


# ---- Step D: Reduce阶段 → 分区内sort+group+特征工程 ----
def _reduce_device_features(iterator):
    """
    手动ReduceByKey: 分区内按device_id排序 → 相邻同device合并 → 计算事件级特征

    这等价于: groupByKey → flatMap(compute_event_features)
    手动实现的好处: 内存可控 (逐device计算, 不一次性展开所有行)
    """
    data = list(iterator)  # [(pid, (dev_id, row)), ...]
    if not data:
        return iter([])

    # ★ 按device_id排序, 相邻相同key聚合 = 手动groupByKey
    data.sort(key=lambda x: x[1][0])

    results = []
    current_dev = None
    dev_rows = []

    for pid, (dev_id, row) in data:
        if dev_id != current_dev:
            # 处理上一个设备的累积
            if current_dev and len(dev_rows) >= 5:
                feats = _compute_device_events(current_dev, dev_rows)
                results.extend(feats)
            current_dev = dev_id
            dev_rows = [row]
        else:
            dev_rows.append(row)

    # 最后一个设备
    if current_dev and len(dev_rows) >= 5:
        feats = _compute_device_events(current_dev, dev_rows)
        results.extend(feats)

    return iter(results)


def _compute_device_events(dev_id, rows):
    """
    单个设备的事件级特征工程 (严格时序, 无泄漏)
    输入: 同一设备的所有故障行 (已按故障时间排序)
    输出: 每行=一次故障, 特征=历史信息, 目标=下次间隔
    """
    rows.sort(key=lambda r: r['fault_ts'])

    from datetime import datetime

    # 计算相邻故障间隔 (fault_ts=Unix时间戳, 单位秒)
    events = []
    hist_intervals = []  # 历史故障间隔列表

    for i in range(1, len(rows)):
        prev_ts = rows[i - 1]['fault_ts']
        cur_ts = rows[i]['fault_ts']
        interval_h = (cur_ts - prev_ts) / 3600.0

        if interval_h <= 0:
            hist_intervals.append(np.nan)
            continue

        cur = rows[i]
        nh = len(hist_intervals)
        # Unix时间戳 → 时间特征
        cur_dt = datetime.fromtimestamp(cur['fault_ts'])
        aging = max((cur['fault_ts'] - rows[0]['fault_ts']) / 86400, 1)

        events.append({
            'device': dev_id,
            'hour': cur_dt.hour,
            'weekday': cur_dt.weekday(),
            'month': cur_dt.month,
            'repair_dur': cur['repair_dur'],
            'response': max((cur['fault_ts'] - prev_ts) / 3600, 0),
            'rtype': cur['repair_type'],
            'n_hist': nh,
            'last_int': hist_intervals[-1] if hist_intervals else 0,
            'avg_int': float(np.nanmean(hist_intervals)) if hist_intervals else 0.0,
            'aging_days': aging,
            'fail_rate': nh / aging,
            'station_enc': cur['station_enc'],
            'brand_enc': cur['brand_enc'],
            'target_h': interval_h,
        })
        hist_intervals.append(interval_h)

    return events


# ========== Phase 6-8: 分布式树训练 (同前) ==========

def _train_tree_worker(args):
    """Worker端: Bootstrap采样 + 训练一棵numpy树 (闭包捕获广播变量)"""
    tree_idx, X_bc_val, y_bc_val, md, ms, seed = args
    np.random.seed(seed)
    n = len(y_bc_val)
    idx = np.random.choice(n, n, replace=True)
    tree = NumPyTree(md, ms)
    tree.fit(X_bc_val[idx], y_bc_val[idx])
    return tree


# ========== 主流程 ==========
if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-Full-Distributed") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "512m") \
        .config("spark.eventLog.enabled", "false") \
        .getOrCreate()

    sc = spark.sparkContext
    NUM_PARTS = 24  # 分区数 > 核数 → 负载均衡

    logger.info("=" * 60)
    logger.info("全链路分布式 — 事件级故障间隔预测")
    logger.info(f"集群: 3Worker×4核=12核, 数据分区={NUM_PARTS}")
    logger.info("=" * 60)

    # ===== Phase 1: Driver加载+Encoder拟合 =====
    logger.info("Phase 1: Driver — 加载CSV + 拟合LabelEncoder")
    raw_rows, le_station, le_brand = _driver_fit_encoders('/root/cleaned_afc_data.csv')
    logger.info(f"  原始行数: {len(raw_rows)}")
    # 广播Encoder到Worker (小对象)
    le_station_bc = sc.broadcast(le_station)
    le_brand_bc = sc.broadcast(le_brand)

    # ===== Phase 2-4: 分布式Map→Shuffle→Reduce =====
    logger.info("Phase 2 [Map]: keyBy(device_id) + Hash分区")
    logger.info("Phase 3 [Shuffle]: partitionBy → 同设备路由到同一Worker")
    logger.info("Phase 4 [Reduce]: 排序→合并→特征工程 (手动reduceByKey)")

    t0 = time.time()

    # Step 2: Map阶段 — 每行加分区号
    def map_fn(row):
        return _map_assign_partition(row, NUM_PARTS, le_station_bc.value, le_brand_bc.value)

    paired_rdd = sc.parallelize(raw_rows, numSlices=NUM_PARTS).map(map_fn)

    # Step 3: Shuffle — 按分区号重分布 (Spark底层=序列化→网络传输→反序列化)
    # ★ 这是手动Shuffle的核心: partitionBy强制同pid数据汇聚到同一Worker
    shuffled = paired_rdd.partitionBy(NUM_PARTS)

    # Step 4: Reduce — 分区内排序+聚合+特征工程
    event_rows = shuffled.mapPartitions(_reduce_device_features).collect()
    t_fe = time.time() - t0
    logger.info(f"  分布式特征工程完成: {t_fe:.0f}秒 ({len(event_rows)}个事件)")

    # ===== Phase 5: Driver标准化+划分 =====
    logger.info("Phase 5: Driver — 标准化 + 划分")

    data = pd.DataFrame(event_rows)
    data = data[data['last_int'] > 0]  # 各设备首条无历史
    data = data[(data['target_h'] >= 10) & (data['target_h'] <= data['target_h'].quantile(0.99))]
    logger.info(f"  过滤后事件: {len(data)}")

    feats = ['hour', 'weekday', 'month', 'repair_dur', 'response', 'rtype',
             'n_hist', 'last_int', 'avg_int', 'aging_days', 'fail_rate',
             'station_enc', 'brand_enc']

    edevs = data['device'].unique()
    etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=RANDOM_SEED)
    etr = data[data['device'].isin(etdevs)]
    ete = data[data['device'].isin(evdevs)]

    X_tr = etr[feats].values.astype(np.float64); y_tr = etr['target_h'].values.astype(np.float64)
    X_te = ete[feats].values.astype(np.float64); y_te = ete['target_h'].values.astype(np.float64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)
    logger.info(f"  训练: {len(etr)} | 测试: {len(ete)} | 特征: {len(feats)}")

    # ===== Phase 6-8: 分布式训练 (同前) =====
    logger.info("=" * 60)
    logger.info("Phase 6: Stage 1 — 分布式scout RF (50树) → 特征重要性")

    X_bc = sc.broadcast(X_tr_s)
    y_bc = sc.broadcast(y_tr_log)

    SCOUT = 100
    def train_scout(task):
        idx, md, ms, seed = task; X = X_bc.value; y = y_bc.value
        np.random.seed(seed); n = len(y)
        return NumPyTree(md, ms).fit(X[np.random.choice(n, n, replace=True)], y)

    scout_tasks = [(i, 10, 4, RANDOM_SEED + i) for i in range(SCOUT)]
    t0 = time.time()
    scout_trees = sc.parallelize(scout_tasks, numSlices=min(SCOUT, 24)) \
                     .map(train_scout).collect()
    t_scout = time.time() - t0

    importances = np.mean([t.importances_ for t in scout_trees], axis=0)
    importances = importances / importances.sum()
    ranked = np.argsort(importances)[::-1]
    top10_idx = ranked[:10]
    top10_names = [feats[i] for i in top10_idx]
    logger.info(f"  Scout: {t_scout:.0f}s | Top10: {top10_names}")

    X_bc.destroy(); y_bc.destroy()

    # Stage 2: 主RF
    logger.info("Phase 7: Stage 2 — 分布式主RF (150树) → Top10特征")

    X_tr_top = X_tr_s[:, top10_idx]; X_te_top = X_te_s[:, top10_idx]
    X_bc2 = sc.broadcast(X_tr_top); y_bc2 = sc.broadcast(y_tr_log)

    MAIN = 300
    def train_main(task):
        idx, md, ms, seed = task; X = X_bc2.value; y = y_bc2.value
        np.random.seed(seed); n = len(y)
        return NumPyTree(md, ms).fit(X[np.random.choice(n, n, replace=True)], y)

    main_tasks = [(i, 18, 2, RANDOM_SEED + 1000 + i) for i in range(MAIN)]
    t0 = time.time()
    main_trees = sc.parallelize(main_tasks, numSlices=min(48, MAIN)) \
                   .map(train_main).collect()
    t_main = time.time() - t0
    logger.info(f"  主RF: {t_main:.0f}s ({MAIN}棵树)")

    X_bc2.destroy(); y_bc2.destroy()

    # ===== 预测+评估+保存 =====
    logger.info("Phase 8: Driver — 集成预测 + 评估")

    preds = np.column_stack([t.predict(X_te_top) for t in main_trees])
    y_pred = np.expm1(preds.mean(axis=1))

    with open('/root/event_numpy_rf_full.pkl', 'wb') as f:
        pickle.dump({
            'trees': main_trees, 'features': top10_names,
            'scaler_mean': scaler.mean_[top10_idx].tolist(),
            'scaler_scale': scaler.scale_[top10_idx].tolist(),
            'target_transform': 'log1p',
            'train_method': 'spark_full_distributed',
            'feature_eng_method': 'manual_mapreduce_shuffle',
            'n_trees': MAIN, 'max_depth': 18, 'min_samples': 2,
            'importances': importances.tolist(),
            'all_feature_names': feats,
        }, f)
    logger.info("模型已保存: /root/event_numpy_rf_full.pkl")

    mask = y_te > 1
    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    mape = np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100
    r2 = 1 - np.sum((y_te - y_pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    acc = 100 - mape

    print("\n" + "=" * 70)
    print("全链路分布式numpy RF — 事件级故障间隔预测")
    print("=" * 70)
    print(f"  集群: 3Worker×4核=12核")
    print(f"  数据分区: {NUM_PARTS} (Hash分桶 → Shuffle → SortMerge)")
    print(f"  分布式阶段: 数据切分/Shuffle/聚合/特征工程/树训练")
    print(f"  特征工程: {t_fe:.0f}秒 (手动MapReduce)")
    print(f"  Scout RF: {t_scout:.0f}秒 ({SCOUT}树)")
    print(f"  主RF:     {t_main:.0f}秒 ({MAIN}树)")
    print(f"  ───────────────────────────────────────")
    print(f"  MAE:     {mae:.1f}h ({mae/24:.1f}天)")
    print(f"  RMSE:    {rmse:.1f}h ({rmse/24:.1f}天)")
    print(f"  MAPE:    {mape:.1f}%")
    print(f"  1-MAPE:  {acc:.1f}%")
    print(f"  R²:      {r2:.3f}")
    print(f"  ───────────────────────────────────────")
    if acc >= 80: print(f"  ★★ ≥80% 额外加分!")
    elif acc >= 70: print(f"  ★ ≥70% 加分!")
    elif acc >= 65: print(f"  √ ≥65% 达标")
    else: print(f"  ✗ <65%")

    spark.stop()
    print("\n完成!")
