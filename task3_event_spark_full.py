"""
任务3: 事件级回归 — Spark分布式版
===================================
架构: Task2建模数据 → Spark划分/标准化 → sklearn RF训练(Driver) → 评估

分布式体现在: Task2的MapReduce特征工程 + Spark数据加载/广播
树训练用sklearn(Driver), 避免NumPyTree在Worker上的序列化问题
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, os, pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task3-Event-Spark") \
        .config("spark.driver.memory", "1g") \
        .config("spark.eventLog.enabled", "false") \
        .getOrCreate()
    sc = spark.sparkContext

    logger.info("=" * 60)
    logger.info("任务3: 事件级故障间隔预测 (Spark + sklearn)")
    logger.info("=" * 60)

    # ===== Phase 1: 读取Task2建模数据集 =====
    csv_path = '/root/modeling_dataset_event.csv' if os.path.exists('/root/modeling_dataset_event.csv') \
               else 'modeling_dataset_event.csv'
    data = pd.read_csv(csv_path, encoding='utf-8-sig')
    feats = ['故障小时','故障星期','故障月份','维修时长_小时','响应时间_小时','维修类型',
             '历史次数','上次间隔_小时','平均间隔_小时','运行天数','故障频率_次每天',
             '车站编码','品牌编码']
    logger.info(f"Phase 1: 加载Task2数据 — {len(data)}样本, {len(feats)}特征")

    # ===== Phase 2: 划分 + 标准化 =====
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    edevs = data['device'].unique()
    etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=RANDOM_SEED)
    etr = data[data['device'].isin(etdevs)]; ete = data[data['device'].isin(evdevs)]
    X_tr = etr[feats].values.astype(np.float64); y_tr = etr['故障间隔_小时'].values.astype(np.float64)
    X_te = ete[feats].values.astype(np.float64); y_te = ete['故障间隔_小时'].values.astype(np.float64)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)
    logger.info(f"Phase 2: 训练{len(etr)} | 测试{len(ete)}")

    # ===== Phase 3: 分布式scout RF → 特征重要性 =====
    logger.info("Phase 3: 分布式scout RF (100树) → 特征重要性")
    # 广播数据到Worker, 分布式训练scout树
    X_bc = sc.broadcast(X_tr_s); y_bc = sc.broadcast(y_tr_log)

    class NumPyTree:
        def __init__(self, md=10, ms=4): self.md=md; self.ms=ms
        def fit(self,X,y):
            self.nf=X.shape[1]; self.imp=np.zeros(self.nf); self.t=self._g(X,y,0); return self
        def _s(self,X,y):
            n=len(y); bg=1e-12; best=(None,None,0.0)
            nf=max(1,int(np.sqrt(X.shape[1])))
            for f in np.random.choice(X.shape[1],nf,replace=False):
                x=X[:,f]; idx=np.argsort(x); xs,ys=x[idx],y[idx]; cs=np.cumsum(ys)
                tv=np.var(ys)*n
                for i in range(self.ms-1, n-self.ms):
                    if xs[i]==xs[i+1]: continue
                    nl,nr=i+1,n-i-1; sl,sr=cs[i],cs[-1]-cs[i]
                    mse=np.sum((ys[:i+1]-sl/nl)**2)+np.sum((ys[i+1:]-sr/nr)**2)
                    g=tv-mse
                    if g>bg: bg=g; best=(f,(xs[i]+xs[i+1])/2,g)
            return best
        def _g(self,X,y,d):
            if d>=self.md or len(y)<self.ms*2: return np.mean(y)
            f,t,gain=self._s(X,y)
            if f is None: return np.mean(y)
            self.imp[f]+=gain; L=X[:,f]<=t; R=~L
            if L.sum()<self.ms or R.sum()<self.ms: return np.mean(y)
            return {'f':f,'t':t,'L':self._g(X[L],y[L],d+1),'R':self._g(X[R],y[R],d+1)}

    SCOUT = 100
    def train_scout_partition(iterator):
        X=X_bc.value; y=y_bc.value
        for task in iterator:
            idx,md,ms,seed=task; np.random.seed(seed); n=len(y)
            yield NumPyTree(md,ms).fit(X[np.random.choice(n,n,replace=True)],y)

    scout_tasks = [(i,10,4,RANDOM_SEED+i) for i in range(SCOUT)]
    t0=time.time()
    scout_trees = list(sc.parallelize(scout_tasks, numSlices=min(SCOUT,24))
                        .mapPartitions(train_scout_partition).collect())
    importances = np.mean([t.imp for t in scout_trees], axis=0)
    importances = importances/importances.sum()
    ranked = np.argsort(importances)[::-1]; top10_idx = ranked[:10]
    top10_names = [feats[i] for i in top10_idx]
    t_scout = time.time()-t0
    logger.info(f"  Scout: {t_scout:.0f}s | Top10: {top10_names}")
    X_bc.destroy(); y_bc.destroy()

    # ===== Phase 4: sklearn主RF (Driver端) =====
    logger.info("Phase 4: sklearn RF (Driver, 替代NumPyTree避免Worker序列化问题)")
    from sklearn.ensemble import RandomForestRegressor
    X_tr_top = X_tr_s[:, top10_idx].copy(); X_te_top = X_te_s[:, top10_idx].copy()
    MAIN = 300
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=MAIN, max_depth=20, min_samples_leaf=2,
                                min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr_top, y_tr_log)
    t_main = time.time()-t0
    logger.info(f"  sklearn RF: {t_main:.0f}s ({MAIN}树×depth=20)")

    # ===== Phase 5: 评估 =====
    y_pred = np.expm1(rf.predict(X_te_top))
    mask = y_te > 1
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mape = np.mean(np.abs((y_te[mask]-y_pred[mask])/y_te[mask]))*100
    r2 = 1-np.sum((y_te-y_pred)**2)/np.sum((y_te-y_te.mean())**2)
    acc = 100-mape

    print("\n" + "=" * 70)
    print("事件级故障间隔预测 (Spark+sklearn)")
    print("=" * 70)
    print(f"  集群: 3Worker×4核=12核")
    print(f"  Scout: {SCOUT}树(分布式) → Top10特征")
    print(f"  主RF: {MAIN}树×depth=20 (sklearn Driver)")
    print(f"  Scout: {t_scout:.0f}s | 主RF: {t_main:.0f}s")
    print(f"  ─────────────────────────────")
    print(f"  MAE:  {mae:.1f}h ({mae/24:.1f}天)")
    print(f"  RMSE: {rmse:.1f}h ({rmse/24:.1f}天)")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  1-MAPE: {acc:.1f}%")
    print(f"  R2:   {r2:.3f}")
    if acc >= 80: print("  ** >=80%!")
    elif acc >= 70: print("  * >=70%!")
    elif acc >= 65: print("  OK >=65%")
    else: print(f"  <65%")

    # 保存模型
    with open('/root/event_sklearn_rf.pkl', 'wb') as f:
        pickle.dump({'model': rf, 'features': top10_names,
                     'scaler_mean': scaler.mean_[top10_idx].tolist(),
                     'scaler_scale': scaler.scale_[top10_idx].tolist()}, f)
    logger.info("模型已保存: /root/event_sklearn_rf.pkl")
    spark.stop(); print("\n完成!")
