"""
任务3 纯numpy随机森林（分布式版）
===================================
- 数据预处理: Spark + pandas
- 模型: 手写numpy RF (决策树分裂+Bagging集成)
- 训练: Driver端多核 (暂不拆分到Worker)
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, pickle as pkl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===================== 纯numpy决策树 =====================
class NumPyTree:
    def __init__(self, max_depth=12, min_samples=3):
        self.max_depth = max_depth; self.min_samples = min_samples

    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0); return self

    def _best_split(self, X, y):
        n = len(y); best_gain = 1e-10; best = (None, None)
        n_feats = max(1, int(np.sqrt(X.shape[1])))
        for f in np.random.choice(X.shape[1], n_feats, replace=False):
            x = X[:, f]; idx = np.argsort(x)
            xs, ys = x[idx], y[idx]
            cs = np.cumsum(ys)
            for i in range(self.min_samples, n - self.min_samples):
                if xs[i] == xs[i+1]: continue
                nl, nr = i + 1, n - i - 1
                sl, sr = cs[i], cs[-1] - cs[i]
                ml, mr = sl / nl, sr / nr
                mse = np.sum((ys[:i+1] - ml)**2) + np.sum((ys[i+1:] - mr)**2)
                gain = np.var(ys) * n - mse
                if gain > best_gain:
                    best_gain = gain
                    best = (f, (xs[i] + xs[i+1]) / 2.0)
        return best[0], best[1]

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples * 2:
            return np.mean(y)  # 叶节点 = 预测均值
        f, t = self._best_split(X, y)
        if f is None: return np.mean(y)
        left = X[:, f] <= t; right = ~left
        if left.sum() < self.min_samples or right.sum() < self.min_samples:
            return np.mean(y)
        return {'f': f, 't': t,
                'l': self._build(X[left], y[left], depth + 1),
                'r': self._build(X[right], y[right], depth + 1)}

    def _pred_one(self, x):
        node = self.tree_
        while isinstance(node, dict):
            node = node['l'] if x[node['f']] <= node['t'] else node['r']
        return node

    def predict(self, X):
        return np.array([self._pred_one(x) for x in X])

# ===================== 纯numpy随机森林 =====================
class NumPyRF:
    def __init__(self, n_trees=150, max_depth=12, min_samples=3):
        self.n_trees = n_trees; self.max_depth = max_depth; self.min_samples = min_samples

    def fit(self, X, y):
        self.trees_ = []; n = len(y)
        for i in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)  # Bootstrap
            tree = NumPyTree(self.max_depth, self.min_samples).fit(X[idx], y[idx])
            self.trees_.append(tree)
            if (i+1) % 30 == 0: logger.info(f"  树 {i+1}/{self.n_trees}...")
        return self

    def predict(self, X):
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        return preds.mean(axis=1)

# ===================== 主流程 =====================
if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-NumPy-RF") \
        .config("spark.driver.memory", "512m") \
        .config("spark.eventLog.enabled", "false") \
        .getOrCreate()

    logger.info("=" * 60)
    logger.info("纯numpy随机森林 — 设备级故障周期预测")
    logger.info("=" * 60)

    # ===== 1. 数据加载 =====
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
            'station_raw': str(dev_df['车站名称'].iloc[0]),
            'brand_raw': str(dev_df['设备品牌'].iloc[0]),
            'line_raw': str(dev_df['线路编号'].iloc[0]),
            'avg_mtbf_h': float(np.mean(intervals)),
        })

    data = pd.DataFrame(device_rows)
    # LabelEncoder编码类别
    from sklearn.preprocessing import LabelEncoder
    for cat_col in ['station_raw', 'brand_raw', 'line_raw']:
        data[cat_col.replace('_raw','_enc')] = LabelEncoder().fit_transform(data[cat_col].astype(str))

    features = ['n_failures','std_mtbf','min_mtbf','max_mtbf',
                'avg_repair','max_repair','aging_days','fail_rate',
                'station_enc','brand_enc','line_enc']
    target = 'avg_mtbf_h'
    logger.info(f"设备数: {len(data)}")

    # ===== 2. 划分 =====
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    devs = data['device'].unique()
    data['stratum'] = pd.cut(data[target]/24, bins=[0,15,23,30,40,999], labels=[0,1,2,3,4])
    dev_labels = data.groupby('device')['stratum'].first().loc[devs].values
    tdevs, vdevs = train_test_split(devs, test_size=0.2, random_state=42, stratify=dev_labels)

    train = data[data['device'].isin(tdevs)]; test = data[data['device'].isin(vdevs)]
    X_tr = train[features].values.astype(np.float64); y_tr = train[target].values.astype(np.float64)
    X_te = test[features].values.astype(np.float64);  y_te = test[target].values.astype(np.float64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)  # log变换

    # ===== 3. 训练手写RF =====
    logger.info("训练纯numpy随机森林...")
    logger.info("参数: 150树, depth=12, min_samples=3, LabelEncoder编码")

    t0 = time.time()
    rf = NumPyRF(n_trees=150, max_depth=12, min_samples=3)
    rf.fit(X_tr_s, y_tr_log)
    t_train = time.time() - t0

    # 保存模型（150棵树列表 + 元数据）
    import pickle
    with open('/root/numpy_rf_model.pkl', 'wb') as f:
        pickle.dump({
            'trees': rf.trees_,
            'features': features,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'target_transform': 'log1p',
            'n_trees': 150, 'max_depth': 12, 'min_samples': 3,
        }, f)
    logger.info("模型已保存: /root/numpy_rf_model.pkl")

    y_pred = np.expm1(rf.predict(X_te_s))

    # ===== 4. 评估 =====
    mask = y_te > 1
    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred)**2))
    mape = np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100
    r2 = 1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - y_te.mean())**2)
    acc = 100 - mape

    print("\n" + "=" * 70)
    print("纯numpy随机森林 — 结果")
    print("=" * 70)
    print(f"  训练时间: {t_train:.1f}s (100棵树)")
    print(f"  MAE: {mae/24:.1f}天, RMSE: {rmse/24:.1f}天")
    print(f"  MAPE: {mape:.1f}%, 1-MAPE: {acc:.1f}%")
    print(f"  R²: {r2:.3f}")
    print(f"\n对比:")
    print(f"  sklearn RF: 86.5% (4秒, Cython编译)")
    print(f"  MLlib RF:   52.1% (277秒, 分桶近似)")
    print(f"  numpy RF:   {acc:.1f}% ({t_train:.0f}秒, 手写+解释执行)")
    print(f"\n  解释执行开销: {t_train/4:.0f}x (sklearn=4秒)")

    if acc >= 80: print("★★ ≥80%!")
    elif acc >= 70: print("★ ≥70%!")
    elif acc >= 65: print("√ ≥65%")

    spark.stop()
    print("完成!")
