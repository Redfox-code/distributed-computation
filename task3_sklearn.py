"""
任务3：设备级 + 事件级 故障间隔回归预测
========================================
使用Task2输出的 modeling_dataset_event.csv
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging, time, os, pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)

for f in fm.fontManager.ttflist:
    if 'SimHei' in f.name:
        plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
        break

FIGURE_DIR = 'figures_task3'; os.makedirs(FIGURE_DIR, exist_ok=True)

logger.info("=" * 60)
logger.info("任务3: 故障间隔回归预测")
logger.info("=" * 60)

# ===== 1. 加载数据 =====
pdf = pd.read_csv('cleaned_afc_data.csv', encoding='utf-8-sig')
for c in ['故障时间', '维修开始时间', '维修完成时间']:
    pdf[c] = pd.to_datetime(pdf[c], errors='coerce')

# ===== 2. 设备级特征工程 =====
logger.info("Step 1: 设备级特征工程")
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
        'device': dev_id, 'n_failures': n,
        'std_mtbf_h': np.std(intervals), 'min_mtbf_h': np.min(intervals),
        'max_mtbf_h': np.max(intervals), 'avg_repair_h': np.mean(repairs),
        'max_repair_h': np.max(repairs), 'aging_days': aging,
        'fail_rate_monthly': n / aging * 30,
        'station': str(dev_df['车站名称'].iloc[0]),
        'brand': str(dev_df['设备品牌'].iloc[0]),
        'line': str(dev_df['线路编号'].iloc[0]),
        'avg_mtbf_h': np.mean(intervals),
    })

data = pd.DataFrame(device_rows)
target_col = 'avg_mtbf_h'
data['stratum'] = pd.cut(data[target_col]/24, bins=[0,15,23,30,40,999], labels=False)
for c in ['station', 'brand', 'line']:
    data[c + '_enc'] = LabelEncoder().fit_transform(data[c])

features = ['n_failures','std_mtbf_h','min_mtbf_h','max_mtbf_h',
            'avg_repair_h','max_repair_h','aging_days','fail_rate_monthly',
            'station_enc','brand_enc','line_enc']

devs = data['device'].unique()
dev_labels = data.groupby('device')['stratum'].first().loc[devs].values
tdevs, vdevs = train_test_split(devs, test_size=0.2, random_state=42, stratify=dev_labels)
train = data[data['device'].isin(tdevs)]; test = data[data['device'].isin(vdevs)]
X_tr = train[features].values.astype(np.float64); y_tr = train[target_col].values.astype(np.float64)
X_te = test[features].values.astype(np.float64); y_te = test[target_col].values.astype(np.float64)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
y_tr_log = np.log1p(y_tr)

# ===== 3. 设备级模型训练 =====
logger.info("Step 2: 设备级模型训练")
results, best_preds = {}, {}

t0 = time.time()
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr_log)
best_preds['RF'] = np.expm1(rf.predict(X_te_s))
results['RF'] = time.time() - t0
logger.info(f"  RF: {results['RF']:.1f}s")

# LightGBM
try:
    import lightgbm as lgb
    t0 = time.time()
    lgbr = lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, verbose=-1)
    lgbr.fit(X_tr_s, y_tr_log)
    best_preds['LightGBM'] = np.expm1(lgbr.predict(X_te_s))
    results['LightGBM'] = time.time() - t0
    logger.info(f"  LightGBM: {results['LightGBM']:.1f}s")
except: pass

# XGBoost
try:
    import xgboost as xgb
    t0 = time.time()
    xgbr = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbosity=0)
    xgbr.fit(X_tr_s, y_tr_log)
    best_preds['XGBoost'] = np.expm1(xgbr.predict(X_te_s))
    results['XGBoost'] = time.time() - t0
    logger.info(f"  XGBoost: {results['XGBoost']:.1f}s")
except: pass

# ===== 4. 设备级评估 =====
print("\n" + "=" * 60)
print("设备级回归结果")
print("=" * 60)

def evaluate(name, y_pred):
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mask = y_te > 1
    mape = np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100
    acc  = 100 - mape
    r2   = 1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - y_te.mean())**2)
    print(f"  {name:12s}: MAE={mae/24:.1f}天 RMSE={rmse/24:.1f}天 MAPE={mape:.1f}% 1-MAPE={acc:.1f}% R2={r2:.3f}")
    return acc

best_acc, best_name = 0, ''
for name in best_preds:
    acc = evaluate(name, best_preds[name])
    if acc > best_acc: best_acc, best_name = acc, name

if best_acc >= 80: print(f"\n  最优: {best_name} {best_acc:.1f}% [STAR][STAR] >=80%!")
elif best_acc >= 70: print(f"\n  最优: {best_name} {best_acc:.1f}% [STAR] >=70%!")

# ===== 5. 事件级回归 (使用Task2建模数据集) =====
logger.info("\n" + "=" * 60)
logger.info("Step 3: 事件级回归 (Task2建模数据集)")
logger.info("=" * 60)

evt = pd.read_csv('modeling_dataset_event.csv', encoding='utf-8-sig')
# 原版13特征
evt_feats = ['故障小时','故障星期','故障月份','维修时长_小时','响应时间_小时','维修类型',
             '历史次数','上次间隔_小时','平均间隔_小时','运行天数','故障频率_次每天',
             '车站编码','品牌编码']
logger.info(f"事件级样本: {len(evt)} | 特征: {len(evt_feats)}")

edevs = evt['device'].unique()
etdevs, evdevs = train_test_split(edevs, test_size=0.2, random_state=42)
etr = evt[evt['device'].isin(etdevs)]; ete = evt[evt['device'].isin(evdevs)]
X_etr = etr[evt_feats].values.astype(np.float64); y_etr = etr['故障间隔_小时'].values.astype(np.float64)
X_ete = ete[evt_feats].values.astype(np.float64); y_ete = ete['故障间隔_小时'].values.astype(np.float64)

scaler_evt = StandardScaler()
X_etr_s = scaler_evt.fit_transform(X_etr); X_ete_s = scaler_evt.transform(X_ete)

# SelectFromModel Top10
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                      max_features=10, threshold=-np.inf)
sfm.fit(X_etr_s, np.log1p(y_etr))
top10_mask = sfm.get_support()
top10_feats = [evt_feats[i] for i in range(len(evt_feats)) if top10_mask[i]]
X_etr_s10 = X_etr_s[:, top10_mask]; X_ete_s10 = X_ete_s[:, top10_mask]
logger.info(f"  Top10: {top10_feats}")

rf_evt = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                min_samples_split=5, random_state=42, n_jobs=-1)
rf_evt.fit(X_etr_s10, np.log1p(y_etr))
y_pred_evt = np.expm1(rf_evt.predict(X_ete_s10))

mask_evt = y_ete > 1
mape_evt = np.mean(np.abs((y_ete[mask_evt] - y_pred_evt[mask_evt]) / y_ete[mask_evt])) * 100
mae_evt  = mean_absolute_error(y_ete, y_pred_evt)
rmse_evt = np.sqrt(mean_squared_error(y_ete, y_pred_evt))
r2_evt = 1 - np.sum((y_ete - y_pred_evt)**2) / np.sum((y_ete - y_ete.mean())**2)
acc_evt = 100 - mape_evt
logger.info(f"  事件级RF: MAE={mae_evt:.1f}h RMSE={rmse_evt:.1f}h MAPE={mape_evt:.1f}% 1-MAPE={acc_evt:.1f}% R2={r2_evt:.3f}")

print(f"\n{'='*60}")
print(f"设备级 vs 事件级")
print(f"{'='*60}")
print(f"  设备级 ({best_name}):     1-MAPE = {best_acc:.1f}%")
print(f"  事件级 (RF+Task2): 1-MAPE = {acc_evt:.1f}%")

# ===== 6. 保存模型 =====
logger.info("Step 4: 保存模型")
with open('best_regressor.pkl', 'wb') as f: pickle.dump(rf, f)
with open('event_regressor.pkl', 'wb') as f: pickle.dump(rf_evt, f)
logger.info("  已保存: best_regressor.pkl, event_regressor.pkl")

# ===== 7. 图表 =====
# Fig1: 设备级模型对比
fig, ax = plt.subplots(figsize=(8, 5))
names = list(best_preds.keys())
accs = []
for name in names:
    mask = y_te > 1
    m = np.mean(np.abs((y_te[mask]-best_preds[name][mask])/y_te[mask]))*100
    accs.append(100-m)
colors = ['#2ecc71' if a == max(accs) else '#3498db' for a in accs]
ax.barh(names, accs, color=colors, edgecolor='white')
for i,(n,a) in enumerate(zip(names,accs)): ax.text(a+0.5, i, f'{a:.1f}%', va='center')
ax.set_xlabel('1-MAPE (%)', fontsize=12)
ax.set_title('设备级模型对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig1_设备级模型对比.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig2: 设备级预测vs实际
fig, ax = plt.subplots(figsize=(7, 7))
best_pred = best_preds[best_name]
ax.scatter(y_te/24, best_pred/24, alpha=0.5, s=12, c='steelblue', edgecolors='none')
mx = max(y_te.max(), best_pred.max())/24*1.05
ax.plot([0,mx],[0,mx],'r--',lw=1.5,label='Perfect')
ax.set_xlim(0,mx); ax.set_ylim(0,mx)
ax.set_xlabel('Actual MTBF (days)', fontsize=12)
ax.set_ylabel('Predicted MTBF (days)', fontsize=12)
ax.set_title(f'{best_name}: Pred vs Actual ({best_acc:.1f}%)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig2_设备级预测vs实际.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig3: 事件级预测vs实际
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_ete/24, y_pred_evt/24, alpha=0.3, s=6, c='steelblue', edgecolors='none')
mx2 = max(y_ete.max(), y_pred_evt.max())/24*1.05
ax.plot([0,mx2],[0,mx2],'r--',lw=1.5,label='Perfect')
ax.set_xlim(0,mx2); ax.set_ylim(0,mx2)
ax.set_xlabel('Actual Interval (days)', fontsize=12)
ax.set_ylabel('Predicted Interval (days)', fontsize=12)
ax.set_title(f'Event-level: Pred vs Actual ({acc_evt:.1f}%)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig3_事件级预测vs实际.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig4: 特征重要性
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
imp_rf = pd.Series(rf.feature_importances_, index=features).sort_values().tail(8)
axes[0].barh(range(len(imp_rf)), imp_rf.values, color='steelblue')
axes[0].set_yticks(range(len(imp_rf))); axes[0].set_yticklabels(imp_rf.index, fontsize=9)
axes[0].set_title('Device-level RF Importance', fontsize=14)
imp_evt = pd.Series(rf_evt.feature_importances_, index=top10_feats).sort_values()
axes[1].barh(range(len(imp_evt)), imp_evt.values, color='coral')
axes[1].set_yticks(range(len(imp_evt))); axes[1].set_yticklabels(imp_evt.index, fontsize=9)
axes[1].set_title('Event-level RF Top10 Importance', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_特征重要性.png'), dpi=150, bbox_inches='tight')
plt.close()

logger.info(f"  图表: {FIGURE_DIR}/ (4张)")
print("\n完成!")
