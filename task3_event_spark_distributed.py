"""
事件级回归 — 分布式numpy RF (Spark broadcast+map)
===================================================
架构: Driver特征工程 → broadcast(X,y) → parallelize(trees).map(train)
       → collect树 → Driver集成预测

用法: spark-submit task3_event_spark_distributed.py
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42

# ===== numpy决策树 =====
class NumPyTree:
    def __init__(self, max_depth=15, min_samples=3):
        self.max_depth = max_depth; self.min_samples = min_samples
    def fit(self, X, y):
        self.n_features_ = X.shape[1]; self.importances_ = np.zeros(self.n_features_)
        self.tree_ = self._build(X, y, 0); return self
    def _best_split(self, X, y):
        n = len(y); best_gain = 1e-12; best = (None, None, 0.0)
        n_feats = max(1, int(np.sqrt(X.shape[1])))
        for f in np.random.choice(X.shape[1], n_feats, replace=False):
            x = X[:, f]; idx = np.argsort(x); xs, ys = x[idx], y[idx]; cs = np.cumsum(ys)
            tv = np.var(ys) * n
            for i in range(self.min_samples, n - self.min_samples):
                if xs[i] == xs[i+1]: continue
                nl, nr = i+1, n-i-1; sl, sr = cs[i], cs[-1]-cs[i]
                mse = np.sum((ys[:i+1]-sl/nl)**2) + np.sum((ys[i+1:]-sr/nr)**2)
                g = tv - mse
                if g > best_gain: best_gain = g; best = (f, (xs[i]+xs[i+1])/2.0, g)
        return best
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples * 2: return np.mean(y)
        f, t, gain = self._best_split(X, y)
        if f is None: return np.mean(y)
        self.importances_[f] += gain
        left = X[:, f] <= t; right = ~left
        if left.sum() < self.min_samples or right.sum() < self.min_samples: return np.mean(y)
        return {'f': f, 't': t, 'L': self._build(X[left], y[left], depth+1),
                'R': self._build(X[right], y[right], depth+1)}
    def predict(self, X):
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self.tree_
            while isinstance(node, dict): node = node['L'] if x[node['f']] <= node['t'] else node['R']
            out[i] = node
        return out

# ===== 主流程 =====
if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-Event-Distributed-RF") \
        .config("spark.driver.memory", "1g").config("spark.executor.memory", "512m") \
        .config("spark.eventLog.enabled", "false").getOrCreate()
    sc = spark.sparkContext
    logger.info("="*60); logger.info("事件级故障间隔预测 — 分布式numpy RF"); logger.info("="*60)

    # ---- 1. 特征工程 (Driver) ----
    logger.info("Step 1: 事件级特征工程 (Driver)")
    pdf = pd.read_csv('/root/cleaned_afc_data.csv', encoding='utf-8-sig')
    for c in ['故障时间','维修开始时间','维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')
    pdf = pdf.sort_values(['设备编号','故障时间'])

    from sklearn.preprocessing import LabelEncoder
    le_s = LabelEncoder().fit(pdf['车站名称'].astype(str))
    le_b = LabelEncoder().fit(pdf['设备品牌'].astype(str))

    rows = []
    for dev_id, dev_df in pdf.groupby('设备编号'):
        dev_df = dev_df.sort_values('故障时间').reset_index(drop=True)
        intervals = dev_df['故障时间'].diff().dt.total_seconds() / 3600
        hist = []
        for i in range(1, len(dev_df)):
            tgt = intervals.iloc[i]
            if np.isnan(tgt) or tgt <= 0: continue
            cur = dev_df.iloc[i]; nh = len(hist)
            aging = max((cur['故障时间']-dev_df['故障时间'].iloc[0]).total_seconds()/86400, 1)
            rows.append({
                'device': dev_id,
                'hour': cur['故障时间'].hour, 'weekday': cur['故障时间'].dayofweek,
                'month': cur['故障时间'].month, 'repair_dur': cur['维修时长_小时'],
                'response': max((cur['维修开始时间']-cur['故障时间']).total_seconds()/3600, 0),
                'rtype': 1 if cur['维修类型']=='CBM' else 0,
                'n_hist': nh, 'last_int': hist[-1] if hist else 0,
                'avg_int': np.mean(hist) if hist else 0,
                'aging_days': aging, 'fail_rate': nh/aging,
                'station_enc': le_s.transform([str(cur['车站名称'])])[0],
                'brand_enc': le_b.transform([str(cur['设备品牌'])])[0],
                'target_h': tgt,
            })
            hist.append(tgt)

    data = pd.DataFrame(rows)
    data = data[data['last_int'] > 0]
    data = data[(data['target_h']>=10) & (data['target_h']<=data['target_h'].quantile(0.99))]
    logger.info(f"  事件数: {len(data)}")

    feats = ['hour','weekday','month','repair_dur','response','rtype',
             'n_hist','last_int','avg_int','aging_days','fail_rate',
             'station_enc','brand_enc']

    # ---- 2. 划分+标准化 ----
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    edevs = data['device'].unique()
    etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=RANDOM_SEED)
    etr = data[data['device'].isin(etdevs)]; ete = data[data['device'].isin(evdevs)]
    X_tr = etr[feats].values.astype(np.float64); y_tr = etr['target_h'].values.astype(np.float64)
    X_te = ete[feats].values.astype(np.float64); y_te = ete['target_h'].values.astype(np.float64)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)
    logger.info(f"  训练: {len(etr)} | 测试: {len(ete)}")

    # ---- 3. Stage 1: 分布式scout RF ----
    logger.info("Stage 1: 分布式scout RF (100树×depth=10) → 特征重要性")
    X_bc = sc.broadcast(X_tr_s); y_bc = sc.broadcast(y_tr_log)

    SCOUT = 100
    def train_scout(task):
        idx, md, ms, seed = task; X = X_bc.value; y = y_bc.value
        np.random.seed(seed); n = len(y)
        return NumPyTree(md, ms).fit(X[np.random.choice(n, n, replace=True)], y)

    t0 = time.time()
    scout_trees = sc.parallelize([(i, 10, 4, RANDOM_SEED+i) for i in range(SCOUT)], 24) \
                    .map(train_scout).collect()
    t_scout = time.time() - t0

    importances = np.mean([t.importances_ for t in scout_trees], axis=0)
    importances = importances / importances.sum()
    ranked = np.argsort(importances)[::-1]; top10_idx = ranked[:10]
    top10_names = [feats[i] for i in top10_idx]
    logger.info(f"  Scout: {t_scout:.0f}s | Top10: {top10_names}")
    X_bc.destroy(); y_bc.destroy()

    # ---- 4. Stage 2: 分布式主RF ----
    logger.info("Stage 2: 分布式主RF (300树×depth=18) → Top10特征")
    X_tr_top = X_tr_s[:, top10_idx]; X_te_top = X_te_s[:, top10_idx]
    X_bc2 = sc.broadcast(X_tr_top); y_bc2 = sc.broadcast(y_tr_log)

    MAIN = 300
    def train_main(task):
        idx, md, ms, seed = task; X = X_bc2.value; y = y_bc2.value
        np.random.seed(seed); n = len(y)
        return NumPyTree(md, ms).fit(X[np.random.choice(n, n, replace=True)], y)

    t0 = time.time()
    main_trees = sc.parallelize([(i, 18, 2, RANDOM_SEED+1000+i) for i in range(MAIN)], 48) \
                   .map(train_main).collect()
    t_main = time.time() - t0
    logger.info(f"  主RF: {t_main:.0f}s ({MAIN}棵树)")
    X_bc2.destroy(); y_bc2.destroy()

    # ---- 5. 预测+评估 ----
    logger.info("Stage 3: 集成预测 (Driver)")
    preds = np.column_stack([t.predict(X_te_top) for t in main_trees])
    y_pred = np.expm1(preds.mean(axis=1))

    with open('/root/event_numpy_rf.pkl', 'wb') as f:
        pickle.dump({
            'trees': main_trees, 'features': top10_names,
            'scaler_mean': scaler.mean_[top10_idx].tolist(),
            'scaler_scale': scaler.scale_[top10_idx].tolist(),
            'target_transform': 'log1p',
        }, f)
    logger.info("模型已保存: /root/event_numpy_rf.pkl")

    mask = y_te > 1
    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred)**2))
    mape = np.mean(np.abs((y_te[mask]-y_pred[mask])/y_te[mask]))*100
    r2 = 1-np.sum((y_te-y_pred)**2)/np.sum((y_te-y_te.mean())**2)
    acc = 100-mape

    print("\n"+"="*70)
    print("分布式numpy RF (Spark) — 事件级故障间隔预测")
    print("="*70)
    print(f"  集群: 3Worker×4核=12核")
    print(f"  Scout: {SCOUT}树×depth=10 ({t_scout:.0f}s)")
    print(f"  主RF: {MAIN}树×depth=18 ({t_main:.0f}s)")
    print(f"  ─────────────────────────────")
    print(f"  MAE: {mae:.1f}h ({mae/24:.1f}天)  RMSE: {rmse:.1f}h ({rmse/24:.1f}天)")
    print(f"  MAPE: {mape:.1f}%  1-MAPE: {acc:.1f}%  R2: {r2:.3f}")
    if acc >= 80: print("  [STAR][STAR] >=80%!")
    elif acc >= 70: print("  [STAR] >=70%!")
    elif acc >= 65: print("  [PASS] >=65%")
    else: print(f"  [FAIL] <65%")

    spark.stop(); print("\n完成!")
