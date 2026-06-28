"""
任务2：事件级特征工程 + 筛选 + 不平衡处理 (Spark分布式版)
==========================================================
架构: Driver加载CSV → 广播Encoder → 分布式MapReduce特征工程
      → Driver筛选(Pearson/MI/VIF) → SMOTE → 轻量RF验证 → 图表输出

用法: spark-submit task2_event_spark.py
"""
from pyspark.sql import SparkSession
import pandas as pd, numpy as np
import logging, time, os, json, warnings, hashlib
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)

FIGURE_DIR = 'figures_task2'; os.makedirs(FIGURE_DIR, exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    if 'SimHei' in f.name:
        plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
        break

# ---- Phase 1 ----
def driver_prepare(csv_path):
    pdf = pd.read_csv(csv_path, encoding='utf-8-sig')
    from sklearn.preprocessing import LabelEncoder
    le_s = LabelEncoder().fit(pdf['车站名称'].astype(str))
    le_b = LabelEncoder().fit(pdf['设备品牌'].astype(str))
    le_l = LabelEncoder().fit(pdf['线路编号'].astype(str))
    for c in ['故障时间','维修开始时间','维修完成时间']:
        pdf[c] = pd.to_datetime(pdf[c], errors='coerce')
    rows = []
    for _, r in pdf.iterrows():
        ft, rst = r['故障时间'], r['维修开始时间']
        rows.append({
            'device': str(r['设备编号']),
            'fault_ts': ft.timestamp() if pd.notna(ft) else 0.0,
            'fault_hour': int(ft.hour) if pd.notna(ft) else 0,
            'fault_weekday': int(ft.dayofweek) if pd.notna(ft) else 0,
            'fault_month': int(ft.month) if pd.notna(ft) else 0,
            'fault_quarter': (ft.month-1)//3+1 if pd.notna(ft) else 0,
            'repair_start_ts': rst.timestamp() if pd.notna(rst) else 0.0,
            'repair_dur': float(r['维修时长_小时']) if pd.notna(r['维修时长_小时']) else 0.0,
            'repair_type': str(r['维修类型']) if pd.notna(r['维修类型']) else '',
            'station': str(r['车站名称']), 'brand': str(r['设备品牌']),
            'line': str(r['线路编号']),
        })
    return rows, le_s, le_b, le_l

# ---- Phase 2-4 ----
def _hash_partition(dev, n): return int(hashlib.md5(dev.encode()).hexdigest(),16) % n

def _map_assign(row, np, le_s, le_b, le_l):
    pid = _hash_partition(row['device'], np)
    enc = {'device':row['device'],'fault_ts':float(row['fault_ts']),
        'fault_hour':int(row['fault_hour']),'fault_weekday':int(row['fault_weekday']),
        'fault_month':int(row['fault_month']),'fault_quarter':int(row['fault_quarter']),
        'repair_start_ts':float(row['repair_start_ts']),'repair_dur':float(row['repair_dur']),
        'repair_type':1 if 'CBM' in row['repair_type'].upper() else 0,
        'station_enc':le_s.transform([row['station']])[0],
        'brand_enc':le_b.transform([row['brand']])[0],
        'line_enc':le_l.transform([row['line']])[0]}
    return (pid, (row['device'], enc))

def _reduce_features(iterator):
    data = list(iterator)
    if not data: return iter([])
    data.sort(key=lambda x: x[1][0])
    res, cur_d, rows = [], None, []
    for _, (dev, row) in data:
        if dev != cur_d:
            if cur_d and len(rows) >= 2: res.extend(_compute_features(cur_d, rows))
            cur_d, rows = dev, [row]
        else: rows.append(row)
    if cur_d and len(rows) >= 2: res.extend(_compute_features(cur_d, rows))
    return iter(res)

def _compute_features(dev_id, rows):
    rows.sort(key=lambda r: r['fault_ts'])
    events, hist = [], []
    for i in range(1, len(rows)):
        interval_h = (rows[i]['fault_ts'] - rows[i-1]['fault_ts'])/3600.0
        if interval_h <= 0: continue
        cur, nh = rows[i], len(hist)
        aging = max((cur['fault_ts']-rows[0]['fault_ts'])/86400, 1)
        is_wknd = 1 if cur['fault_weekday'] >= 5 else 0
        is_peak = 1 if (cur['fault_weekday']<5 and cur['fault_hour'] in [7,8,9,17,18,19]) else 0
        resp = max((cur['repair_start_ts']-cur['fault_ts'])/3600, 0)
        avg_int = float(np.mean(hist)) if hist else 0.0
        std_int = float(np.std(hist)) if len(hist)>=2 else 0.0
        last_int = hist[-1] if hist else 0.0
        avg3 = float(np.mean(hist[-3:])) if len(hist)>=3 else (avg_int if hist else 0.0)
        trend = last_int/(avg_int+1e-8) if hist else 1.0
        trend3 = avg3/(float(np.mean(hist[:-3]))+1e-8) if len(hist)>=6 else 1.0
        frate = nh/aging
        accel = (nh/max(aging-365,1))/((nh+1e-8)/aging) if aging>365 and nh>=2 else 1.0
        rh = [rows[j]['repair_dur'] for j in range(1,i)]
        avg_rep = float(np.mean(rh)) if rh else cur['repair_dur']
        rsp_h = [max((rows[j]['repair_start_ts']-rows[j]['fault_ts'])/3600,0) for j in range(1,i)]
        rsp_trend = resp - (float(np.mean(rsp_h)) if rsp_h else resp)

        events.append({'device':dev_id,
            '故障小时':cur['fault_hour'],'故障星期':cur['fault_weekday'],
            '故障月份':cur['fault_month'],'故障季度':cur['fault_quarter'],
            '是否周末':is_wknd,'是否工作日高峰':is_peak,
            '维修时长_小时':cur['repair_dur'],'响应时间_小时':resp,
            '维修类型':cur['repair_type'],'历史次数':nh,
            '上次间隔_小时':last_int,'平均间隔_小时':avg_int,
            '间隔标准差_小时':std_int,'近期趋势_比值':trend,
            '最近3次平均_小时':avg3,'最近3次趋势':trend3,
            '运行天数':aging,'故障频率_次每天':frate,'故障加速比':accel,
            '平均维修时长_小时':avg_rep,'响应时间趋势':rsp_trend,
            '车站编码':cur['station_enc'],'品牌编码':cur['brand_enc'],
            '线路编码':cur['line_enc'],'故障间隔_小时':interval_h})
        hist.append(interval_h)
    return events

# ===== Main =====
if __name__ == '__main__':
    spark = SparkSession.builder.appName("Task2-Distributed-FE") \
        .config("spark.driver.memory","1g").config("spark.eventLog.enabled","false").getOrCreate()
    sc = spark.sparkContext; NUM_PARTS = 24

    logger.info("="*50); logger.info("任务2: 事件级特征工程 (Spark分布式)"); logger.info("="*50)

    # Phase 1
    logger.info("Phase 1: Driver 加载+广播Encoder")
    csv_path = '/root/cleaned_afc_data.csv' if os.path.exists('/root/cleaned_afc_data.csv') else 'cleaned_afc_data.csv'
    raw_rows, le_s, le_b, le_l = driver_prepare(csv_path)
    le_s_bc, le_b_bc, le_l_bc = sc.broadcast(le_s), sc.broadcast(le_b), sc.broadcast(le_l)

    # Phase 2-4
    logger.info("Phase 2-4: Map+Shuffle+Reduce (手动MapReduce)")
    def map_fn(r): return _map_assign(r, NUM_PARTS, le_s_bc.value, le_b_bc.value, le_l_bc.value)
    t0 = time.time()
    event_rows = sc.parallelize(raw_rows, NUM_PARTS).map(map_fn) \
                   .partitionBy(NUM_PARTS).mapPartitions(_reduce_features).collect()
    logger.info(f"  完成: {time.time()-t0:.0f}s ({len(event_rows)}事件)")

    # Phase 5: 过滤+筛选
    logger.info("Phase 5: 过滤 + Pearson/MI/VIF 三重筛选")
    data = pd.DataFrame(event_rows)
    data = data[data['上次间隔_小时']>0]
    q99 = data['故障间隔_小时'].quantile(0.99)
    data = data[(data['故障间隔_小时']>=10)&(data['故障间隔_小时']<=q99)]
    logger.info(f"  过滤后: {len(data)}事件")

    dev_counts = data['device'].value_counts()
    bins, labels = [0,2,5,10,20,1000], ['1-2次','3-5次','6-10次','11-20次','20+次']
    data['频次档'] = pd.cut(data['device'].map(dev_counts), bins=bins, labels=labels)
    freq_dist = data['频次档'].value_counts().sort_index()

    exclude = ['device','故障间隔_小时','频次档']
    all_feats = [c for c in data.columns if c not in exclude]
    X_all, y_all = data[all_feats], data['故障间隔_小时'].values

    # Pearson
    pearson = {}
    for f in all_feats:
        c = np.corrcoef(X_all[f].values, y_all)[0,1]
        pearson[f] = abs(c) if not np.isnan(c) else 0.0
    # MI
    from sklearn.feature_selection import mutual_info_regression
    mi_raw = mutual_info_regression(X_all.values, y_all, random_state=RANDOM_SEED)
    mi = {all_feats[i]:mi_raw[i] for i in range(len(all_feats))}
    # 预选
    pre_sel = [f for f in all_feats if pearson[f]>=0.03 and mi.get(f,0)>=0.001]
    logger.info(f"  预选: {len(all_feats)} → {len(pre_sel)}")

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    kept, dropped = [], []
    rem = list(pre_sel)
    while True:
        try:
            Xv = X_all[rem].values.astype(np.float64)
            vifs = [variance_inflation_factor(Xv,i) for i in range(Xv.shape[1])]
            if max(vifs) > 10:
                idx = vifs.index(max(vifs))
                dropped.append((rem[idx], max(vifs)))
                rem.pop(idx)
            else: kept = rem; break
        except: kept = rem; break

    vif_scores = {}
    if kept:
        Xv = X_all[kept].values.astype(np.float64)
        for i,n in enumerate(kept): vif_scores[n] = round(variance_inflation_factor(Xv,i),1)

    final_feats = kept
    logger.info(f"  VIF后: {len(final_feats)}特征")

    ranked = sorted(pearson.items(), key=lambda x:x[1], reverse=True)
    print(f"\n{'特征':16s} {'|r|':>7s} {'MI':>7s} {'VIF':>6s} 判定")
    print("-"*45)
    for name, corr in ranked:
        mv, vv = mi.get(name,0), vif_scores.get(name,0)
        if name in final_feats: s = f"  {name:16s} {corr:7.4f} {mv:7.4f} {vv:6.1f} OK"
        elif name in [d[0] for d in dropped]: s = f"  {name:16s} {corr:7.4f} {mv:7.4f} {[d[1] for d in dropped if d[0]==name][0]:6.1f} [DROP]VIF"
        else: s = f"  {name:16s} {corr:7.4f} {mv:7.4f} {'-':>6s} [DROP]弱相关"
        print(s)

    # Phase 6: 不平衡处理
    logger.info("Phase 6: 分层划分 + SMOTE")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    devs = data['device'].unique()
    dev_avg = data.groupby('device')['故障间隔_小时'].mean()
    dev_stratum = pd.cut(dev_avg, bins=5, labels=False)
    t_devs, v_devs = train_test_split(devs, test_size=0.2, random_state=RANDOM_SEED,
                                       stratify=dev_stratum.loc[devs].values)
    train = data[data['device'].isin(t_devs)].copy()
    test  = data[data['device'].isin(v_devs)].copy()
    X_tr = train[final_feats].values.astype(np.float64); y_tr = train['故障间隔_小时'].values.astype(np.float64)
    X_te = test[final_feats].values.astype(np.float64); y_te = test['故障间隔_小时'].values.astype(np.float64)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)

    try:
        from imblearn.over_sampling import SMOTE
        y_binned = pd.cut(pd.Series(y_tr), bins=5, labels=False)
        min_cnt = pd.Series(y_binned).value_counts().min()
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(3,min_cnt-1))
        X_sm, yb_sm = smote.fit_resample(X_tr_s, y_binned)
        if len(X_sm) > 2*len(X_tr_s):
            logger.info(f"  SMOTE限制: {len(X_sm)} -> 1.5x")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=3).fit(X_tr_s)
            n_synth = int(len(X_tr_s)*0.5)
            _, idx = nn.kneighbors(X_sm[len(X_tr_s):len(X_tr_s)+n_synth])
            X_tr_sm = np.vstack([X_tr_s, X_sm[len(X_tr_s):len(X_tr_s)+n_synth]])
            y_tr_sm = np.hstack([y_tr, y_tr[idx[:,0]]])
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=3).fit(X_tr_s)
            _, idx = nn.kneighbors(X_sm[len(X_tr_s):])
            X_tr_sm, y_tr_sm = X_sm, np.hstack([y_tr, y_tr[idx[:,0]]])
        logger.info(f"  SMOTE: {len(y_tr)} -> {len(y_tr_sm)}")
    except Exception as e:
        logger.warning(f"  SMOTE失败({e})")
        X_tr_sm, y_tr_sm = X_tr_s, y_tr

    # Phase 7: 验证
    logger.info("Phase 7: 轻量RF验证")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3,
                               random_state=RANDOM_SEED, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_tr_sm, np.log1p(y_tr_sm))
    y_pred = np.expm1(rf.predict(X_te_s))
    t_val = time.time()-t0
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mask = y_te > 1
    mape = np.mean(np.abs((y_te[mask]-y_pred[mask])/y_te[mask]))*100
    r2 = 1-np.sum((y_te-y_pred)**2)/np.sum((y_te-y_te.mean())**2)
    w = test['device'].map({d:1.0/(len(v_devs)+1e-8) for d in v_devs}).values
    wmape = np.sum(w[mask]*np.abs((y_te[mask]-y_pred[mask])/y_te[mask]))/np.sum(w[mask])*100

    print(f"\n{'='*50}"); print(f"  验证: RF200, log1p, {len(final_feats)}特征")
    print(f"  MAE={mae:.1f}h  RMSE={rmse:.1f}h  MAPE={mape:.1f}%  1-MAPE={100-mape:.1f}%")
    print(f"  wMAPE={wmape:.1f}%  R2={r2:.3f}  ({t_val:.0f}s)")
    imps = pd.Series(rf.feature_importances_, index=final_feats).sort_values(ascending=False)
    print(f"  Top5: {' | '.join([f'{n}={v:.3f}' for n,v in imps.head(5).items()])}")

    # Phase 8: 输出+图表
    logger.info("Phase 8: 输出+图表")
    output = pd.concat([train[['device']+final_feats+['故障间隔_小时']],
                        test[['device']+final_feats+['故障间隔_小时']]], axis=0)
    output.to_csv('modeling_dataset_event.csv', index=False, encoding='utf-8-sig')
    with open('feature_metadata_event.json','w',encoding='utf-8') as f:
        json.dump({'features':final_feats,'n_features':len(final_feats),
            'n_samples':len(output),'n_train':len(train),'n_test':len(test),
            'pearson':{k:round(v,4) for k,v in pearson.items()},
            'mutual_info':{k:round(v,4) for k,v in mi.items()},'vif':vif_scores},
            f, ensure_ascii=False, indent=2)

    # 5 charts
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
    # fig1
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(freq_dist)), freq_dist.values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(freq_dist))); ax.set_xticklabels(freq_dist.index, fontsize=10)
    ax.set_ylabel('事件数量',fontsize=12); ax.set_title('设备事件频次分布(不平衡)',fontsize=14,fontweight='bold')
    for i,(l,c) in enumerate(freq_dist.items()): ax.text(i,c+50,f'{c}\n({c/len(data)*100:.1f}%)',ha='center',fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURE_DIR,'fig1_不平衡分布.png'),dpi=150,bbox_inches='tight'); plt.close()

    # fig2
    fig, ax = plt.subplots(figsize=(10,6))
    n15, v15 = [x[0] for x in ranked[:15]], [x[1] for x in ranked[:15]]
    ax.barh(range(len(n15)), v15, color=['#2ecc71' if v>=0.03 else '#e74c3c' for v in v15], edgecolor='white')
    ax.set_yticks(range(len(n15))); ax.set_yticklabels(n15,fontsize=9)
    ax.set_xlabel('|Pearson r|',fontsize=12); ax.axvline(x=0.03,color='red',linestyle='--',alpha=0.5,label='阈值0.03')
    ax.set_title('特征-目标相关性排序',fontsize=14,fontweight='bold'); ax.legend(); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(FIGURE_DIR,'fig2_相关性筛选.png'),dpi=150,bbox_inches='tight'); plt.close()

    # fig3
    fig, ax = plt.subplots(figsize=(8,5))
    t10 = imps.head(10)
    ax.barh(range(len(t10)), t10.values, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(t10))); ax.set_yticklabels(t10.index,fontsize=10)
    ax.set_xlabel('特征重要性',fontsize=12); ax.set_title('RF特征重要性Top10',fontsize=14,fontweight='bold')
    ax.invert_yaxis(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR,'fig3_特征重要性.png'),dpi=150,bbox_inches='tight'); plt.close()

    # fig4
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(y_te/24, y_pred/24, alpha=0.4, s=8, c='steelblue', edgecolors='none')
    mx = max(y_te.max(),y_pred.max())/24*1.05
    ax.plot([0,mx],[0,mx],'r--',lw=1.5,label='完美预测')
    ax.set_xlim(0,mx); ax.set_ylim(0,mx)
    ax.set_xlabel('实际间隔(天)',fontsize=12); ax.set_ylabel('预测间隔(天)',fontsize=12)
    ax.set_title(f'预测vs实际 (1-MAPE={100-mape:.1f}%)',fontsize=14,fontweight='bold'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGURE_DIR,'fig4_预测vs实际.png'),dpi=150,bbox_inches='tight'); plt.close()

    # fig5
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(y_all/24, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=10/24,color='red',linestyle='--',label='下界10h')
    ax.axvline(x=q99/24,color='orange',linestyle='--',label=f'上界P99({q99/24:.0f}天)')
    ax.set_xlabel('故障间隔(天)',fontsize=12); ax.set_ylabel('频次',fontsize=12)
    ax.set_title(f'目标分布(CV={np.std(y_all)/np.mean(y_all)*100:.1f}%)',fontsize=14,fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR,'fig5_目标分布.png'),dpi=150,bbox_inches='tight'); plt.close()

    logger.info(f"  图表: {FIGURE_DIR}/ (5张)")
    spark.stop(); print("\n完成!")
