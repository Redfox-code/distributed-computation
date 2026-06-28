"""
事件级回归 — 纯numpy RF (单机版)
==================================
每次故障一行, 目标=下一次故障间隔(小时)
含: numpy特征重要性 → 动态选Top10 → 主RF训练
"""
import pandas as pd, numpy as np, logging, time, pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)

# ===== numpy树 + RF (带特征重要性) =====
class Tree:
    def __init__(self, md=15, ms=3): self.md=md; self.ms=ms
    def fit(self,X,y):
        self.n_features_ = X.shape[1]
        self.importances_ = np.zeros(self.n_features_)
        self.t=self._g(X,y,0); return self
    def _s(self,X,y):
        n=len(y); bg=1e-12; best=(None,None,0.0)
        nf=max(1,int(np.sqrt(X.shape[1])))
        for f in np.random.choice(X.shape[1],nf,replace=False):
            x=X[:,f]; idx=np.argsort(x); xs,ys=x[idx],y[idx]; cs=np.cumsum(ys)
            tv=np.var(ys)*n
            for i in range(self.ms,n-self.ms):
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
        self.importances_[f] += gain
        L=X[:,f]<=t; R=~L
        if L.sum()<self.ms or R.sum()<self.ms: return np.mean(y)
        return {'f':f,'t':t,'L':self._g(X[L],y[L],d+1),'R':self._g(X[R],y[R],d+1)}
    def predict(self,X):
        out=np.zeros(len(X))
        for i,x in enumerate(X):
            n=self.t
            while isinstance(n,dict): n=n['L'] if x[n['f']]<=n['t'] else n['R']
            out[i]=n
        return out

class RF:
    def __init__(self,nt=150,md=15,ms=3): self.nt=nt; self.md=md; self.ms=ms
    def fit(self,X,y):
        self.ts=[]; n=len(y)
        for i in range(self.nt):
            idx=np.random.choice(n,n,replace=True)
            self.ts.append(Tree(self.md,self.ms).fit(X[idx],y[idx]))
            if (i+1)%50==0: logger.info(f"  树 {i+1}/{self.nt}...")
        return self
    @property
    def feature_importances_(self):
        imp = np.mean([t.importances_ for t in self.ts], axis=0)
        s = imp.sum(); return imp/s if s>0 else imp
    def predict(self,X):
        ps=np.column_stack([t.predict(X) for t in self.ts])
        return ps.mean(axis=1)

# ===== 主流程 =====
if __name__ == '__main__':
    logger.info("="*60); logger.info("事件级故障间隔预测 — 纯numpy RF"); logger.info("="*60)

    pdf = pd.read_csv('cleaned_afc_data.csv', encoding='utf-8-sig')
    for c in ['故障时间','维修开始时间','维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')
    pdf = pdf.sort_values(['设备编号','故障时间'])

    logger.info("Step 1: 事件级特征工程 (严格时序)")
    from sklearn.preprocessing import LabelEncoder
    all_stations = pdf['车站名称'].astype(str); all_brands = pdf['设备品牌'].astype(str)
    le_s = LabelEncoder().fit(all_stations); le_b = LabelEncoder().fit(all_brands)

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
                'month': cur['故障时间'].month,
                'repair_dur': cur['维修时长_小时'],
                'response': max((cur['维修开始时间']-cur['故障时间']).total_seconds()/3600,0),
                'rtype': 1 if cur['维修类型']=='CBM' else 0,
                'n_hist': nh,
                'last_int': hist[-1] if hist else 0,
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
    logger.info(f"事件数: {len(data)}")

    feats = ['hour','weekday','month','repair_dur','response','rtype',
             'n_hist','last_int','avg_int','aging_days','fail_rate',
             'station_enc','brand_enc']

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    edevs = data['device'].unique()
    etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=42)
    etr = data[data['device'].isin(etdevs)]; ete = data[data['device'].isin(evdevs)]
    X_tr = etr[feats].values.astype(np.float64); y_tr = etr['target_h'].values.astype(np.float64)
    X_te = ete[feats].values.astype(np.float64); y_te = ete['target_h'].values.astype(np.float64)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    y_tr_log = np.log1p(y_tr)
    logger.info(f"训练: {len(etr)} | 测试: {len(ete)} | 特征: {len(feats)}")

    # Stage 1: scout RF → 特征重要性 → Top10
    logger.info("Stage 1: scout RF (50树×depth=8) → 特征重要性")
    scout = RF(nt=50, md=8, ms=5)
    scout.fit(X_tr_s, y_tr_log)
    importances = scout.feature_importances_
    ranked = np.argsort(importances)[::-1]
    top10_idx = ranked[:10]
    top10_names = [feats[i] for i in top10_idx]
    logger.info(f"  Top10: {top10_names}")
    for i, idx in enumerate(ranked):
        logger.info(f"    {i+1:2d}. {feats[idx]:12s} = {importances[idx]:.4f}")

    X_tr_top = X_tr_s[:, top10_idx]; X_te_top = X_te_s[:, top10_idx]

    # Stage 2: 主RF
    logger.info("Stage 2: 主RF (150树×depth=15×ms=3)")
    t0 = time.time()
    rf = RF(nt=150, md=15, ms=3)
    rf.fit(X_tr_top, y_tr_log)
    t_train = time.time() - t0

    y_pred = np.expm1(rf.predict(X_te_top))

    # 保存
    with open('event_numpy_rf.pkl', 'wb') as f:
        pickle.dump({
            'trees': rf.ts, 'features': top10_names,
            'scaler_mean': scaler.mean_[top10_idx].tolist(),
            'scaler_scale': scaler.scale_[top10_idx].tolist(),
            'target_transform': 'log1p',
        }, f)
    logger.info("模型已保存: event_numpy_rf.pkl")

    # 评估
    mask = y_te > 1
    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred)**2))
    mape = np.mean(np.abs((y_te[mask]-y_pred[mask])/y_te[mask]))*100
    r2 = 1-np.sum((y_te-y_pred)**2)/np.sum((y_te-y_te.mean())**2)
    acc = 100-mape

    print("\n" + "="*60)
    print(f"事件级 numpy RF: 1-MAPE={acc:.1f}%")
    print(f"MAE={mae:.1f}h, RMSE={rmse:.1f}h, R2={r2:.3f}")
    print(f"训练: 150树×depth=15×ms=3, {t_train:.0f}秒")
    if acc >= 80: print("[STAR][STAR] >=80%!")
    elif acc >= 70: print("[STAR] >=70%!")
    elif acc >= 65: print("[PASS] >=65%")
    else: print(f"[FAIL] <65% (MAPE={mape:.1f}%)")
