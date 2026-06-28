"""
Task2 图表本地生成
读取 chart_data.json → 生成5张PNG到 figures_task2/
"""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体 ----
for f in fm.fontManager.ttflist:
    if 'SimHei' in f.name:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        break

FIGURE_DIR = 'figures_task2'; os.makedirs(FIGURE_DIR, exist_ok=True)

# ---- 加载数据 ----
with open('chart_data.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

freq_dist = d['freq_dist']
ranked = d['pearson_ranked']
imps_list = d['feature_importances']
y_te = np.array(d['y_test_days'])
y_pred = np.array(d['y_pred_days'])
y_all = np.array(d['y_all_days'])
q99 = d['q99_days']
cv = d['cv']
mape = d['mape']

# ---- Fig1: 不平衡分布 ----
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
labels = list(freq_dist.keys())
values = list(freq_dist.values())
total = d['total_events']
ax.bar(range(len(labels)), values, color=colors, edgecolor='white')
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('事件数量', fontsize=12)
ax.set_title('设备事件频次分布（不平衡）', fontsize=14, fontweight='bold')
for i, (l, v) in enumerate(zip(labels, values)):
    ax.text(i, v+50, f'{v}\n({v/total*100:.1f}%)', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig1_不平衡分布.png'), dpi=150, bbox_inches='tight')
plt.close(); print("fig1 OK")

# ---- Fig2: 特征-目标相关性 ----
fig, ax = plt.subplots(figsize=(10, 6))
n15 = [x[0] for x in ranked[:15]]
v15 = [x[1] for x in ranked[:15]]
bar_colors = ['#2ecc71' if v >= 0.03 else '#e74c3c' for v in v15]
ax.barh(range(len(n15)), v15, color=bar_colors, edgecolor='white')
ax.set_yticks(range(len(n15))); ax.set_yticklabels(n15, fontsize=9)
ax.set_xlabel('|Pearson r|', fontsize=12)
ax.axvline(x=0.03, color='red', linestyle='--', alpha=0.5, label='筛选阈值 0.03')
ax.set_title('特征-目标 Pearson 相关性排序', fontsize=14, fontweight='bold')
ax.legend(); ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig2_相关性筛选.png'), dpi=150, bbox_inches='tight')
plt.close(); print("fig2 OK")

# ---- Fig3: 特征重要性 ----
fig, ax = plt.subplots(figsize=(8, 5))
t10_names = [x[0] for x in imps_list[:10]]
t10_vals  = [x[1] for x in imps_list[:10]]
ax.barh(range(len(t10_names)), t10_vals, color='steelblue', edgecolor='white')
ax.set_yticks(range(len(t10_names))); ax.set_yticklabels(t10_names, fontsize=10)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('RF 特征重要性 Top10', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig3_特征重要性.png'), dpi=150, bbox_inches='tight')
plt.close(); print("fig3 OK")

# ---- Fig4: 预测vs实际 ----
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_te, y_pred, alpha=0.4, s=8, c='steelblue', edgecolors='none')
mx = max(y_te.max(), y_pred.max()) * 1.05
ax.plot([0, mx], [0, mx], 'r--', lw=1.5, label='完美预测')
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel('实际间隔（天）', fontsize=12); ax.set_ylabel('预测间隔（天）', fontsize=12)
ax.set_title(f'预测 vs 实际 (1-MAPE={100-mape:.1f}%)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig4_预测vs实际.png'), dpi=150, bbox_inches='tight')
plt.close(); print("fig4 OK")

# ---- Fig5: 目标分布 ----
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_all, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=10/24, color='red', linestyle='--', label='下界 10h')
ax.axvline(x=q99, color='orange', linestyle='--', label=f'上界 P99 ({q99:.0f}天)')
ax.set_xlabel('故障间隔（天）', fontsize=12); ax.set_ylabel('频次', fontsize=12)
ax.set_title(f'目标变量分布 (CV={cv:.1f}%)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig5_目标分布.png'), dpi=150, bbox_inches='tight')
plt.close(); print("fig5 OK")

print(f"\n5张图表已保存到 {FIGURE_DIR}/")
