# 项目进展总结

## 已完成

### 任务1：数据清洗（20分）✅
- 读取 `设备预测数据.xlsx` (29,964行×29列)
- 删除7819条重复行，核心字段0缺失
- 时间字段统一为datetime，业务逻辑异常清零
- 输出 `cleaned_afc_data.csv` (22,145行)
- 脚本: `task1.py` / `task1_spark.py`

### 任务2：特征工程（20分）✅
- 按设备分组按时序计算故障间隔（目标变量）
- 衍生18个特征（时间6 + 维修4 + 设备统计8）
- 分类编码（Label/Frequency/Target Encoding）
- 零方差→相关性→VIF→业务逻辑四步筛选，保留13特征
- 输出 `modeling_dataset.csv` (17,349行×15列)
- 脚本: `task2.py` / `task2_spark.py`

### 任务3：模型构建（60分）

#### 3.1 事件级回归（原始版）
- 每次故障一行，预测单次间隔
- sklearn Stacking集成：**1-MAPE=71.99%**
- 脚本: `task3.py`

#### 3.2 设备级回归（sklearn版）★ 主力
- 每台设备一行，预测平均MTBF
- 分层划分 + SMOTE + log1p变换
- 多模型对比：**MLP(PyTorch) 86.9%**, RF 86.5%, LightGBM 86.4%, XGBoost 85.9%
- 脚本: `task3_sklearn.py`
- 模型: `best_model.pth` (PyTorch), `best_regressor.pkl` (sklearn)

#### 3.3 事件级回归（sklearn优化版）
- RF300 + depth=20 + Top10特征筛选 → **1-MAPE=72.0%**
- 脚本: `task3_sklearn.py` 内事件级部分

#### 3.4 分布式训练探索

**Spark MLlib** (已放弃):
- MLlib原生RF/GBT: 精度52.1%，RF 277s / GBT 811s
- 底层优化(maxBins/subsampling): 速度提升5-7×，精度无改善(52.1%)
- 结论: MLlib分桶近似算法在小数据上信息损失过大

**纯numpy手写RF** ★:
- 设备级: 150树 depth=12 min=3 + LabelEncoder → **86.3%** (283s)
- 事件级: 100树 depth=10 min=5 + Top10特征 → 训练中
- 算法等价sklearn(精确分裂点+Bagging)，精度差0.2%
- 脚本: `task3_spark_final.py` (设备级), `task3_event_numpy.py` (事件级)
- 模型: `numpy_rf_model.pkl` (设备级), `event_numpy_rf.pkl` (事件级-待完成)

### Spark集群部署 ✅
- 1 Master + 3 Workers (4核×768MB每节点, 总共12核2.3GB)
- Spark 3.5.8 + Hadoop 3.3.6 + CentOS 7
- 脚本通过SFTP上传，spark-submit提交
- 工具: `submit_to_cluster.py`

## 当前模型文件

| 文件 | 模型 | 粒度 | 精度 |
|------|------|------|:---:|
| `best_model.pth` | PyTorch MLP | 设备级 | 86.9% |
| `best_regressor.pkl` | sklearn RF | 设备级 | 86.5% |
| `numpy_rf_model.pkl` | numpy手写RF | 设备级 | 86.3% |
| `event_numpy_rf.pkl` | numpy手写RF | 事件级 | 待完成 |

## 关键决策记录

1. **设备级 vs 事件级**：设备级聚合同一设备的多次故障取平均MTBF，比事件级单次预测稳定得多（86.9% vs 72.0%）
2. **不用MLlib**：分桶近似分裂点在小数据上精度损失太大（52%），改用纯numpy手写RF
3. **不用分布式训练**：1500行数据太小，分布式通信开销>计算收益
4. **LabelEncoder vs hash编码**：LabelEncoder保留类别有序性，精度比hash高
5. **事件级回归精度天花板**：单次故障间隔随机性157%，物理上限约72%
6. **numpy手写RF不自动并行**：Python单线程，加CPU核不会加速

## 未完成待办

- [ ] **事件级numpy RF完成训练**：`task3_event_numpy.py`在Master上内存不足，建议本地运行 (`python task3_event_numpy.py`)
- [ ] 更新操作总结文档加入事件级numpy RF结果
- [ ] 清理Master上残留的日志文件
- [ ] 最终git commit + push
- [ ] 作业PPT/报告

## 环境速查

**本地**: Windows 11, Python 3.11, 项目路径 `C:\Users\24924\Desktop\交通大数据与分布式计算\2026最终作业`

**集群Master**: `ssh root@192.168.149.128` (密码123456zz)
- Spark: `/root/software/spark-3.5.8/`
- Python: `/root/anaconda3/envs/pyspark/bin/python`
- 关键文件: `/root/task3_sklearn.py`, `/root/task3_spark_final.py`, `/root/task3_event_numpy.py`

**Git**: `https://github.com/Redfox-code/2026----.git`, 当前在 `event` 分支

---

## 对话上下文压缩摘要 (2026-06-27)

### 1. 主要请求

- **Task 1 (20分)**: AFC维修日志数据清洗
- **Task 2 (20分)**: 特征工程与筛选
- **Task 3 (60分)**: 设备故障间隔回归预测（MAE/RMSE/MAPE，1-MAPE≥65%通过，≥70%加分，≥80%额外加分，需输出.pth模型文件）
- Spark集群部署（3台CentOS7 VM）
- 分布式训练探索
- 纯numpy手写RF实现
- Git版本管理

### 2. 关键技术概念

- **设备级 vs 事件级**: 设备级聚合多故障取平均MTBF（86.9%），事件级预测单次间隔（72%天花板）
- **log1p变换**: 压缩右偏MTBF分布，`np.expm1`还原
- **分层设备划分**: 按设备ID分训练/测试，防数据泄漏
- **SMOTE过采样**: 平衡少数类
- **MLlib vs sklearn vs numpy RF**: MLlib用分桶近似(52%)，sklearn用Cython精确分裂(86.5%)，numpy手写等价sklearn但Python解释执行(86.3%)
- **Spark Standalone集群**: 3 Workers × 4核 × 768MB RAM，共12核2.3GB
- **精确分裂算法**: `np.argsort`预排序 + `np.cumsum`累积和，每棵树O(n log n)
- **LabelEncoder vs hash**: LabelEncoder保留类别有序性，hash()%N破坏信息

### 3. 关键文件

| 文件 | 用途 |
|------|------|
| `task1.py` | 数据清洗 |
| `task2.py` | 特征工程 |
| `task3_sklearn.py` ★ | 最终模型：设备级MLP 86.9% + 事件级RF300 72.0% |
| `task3_spark_final.py` | 设备级纯numpy RF，150树，86.3% |
| `task3_event_numpy.py` | 事件级纯numpy RF，100树，Top10特征 |
| `submit_to_cluster.py` | SSH/SFTP提交工具 |
| `best_model.pth` | PyTorch MLP（设备级86.9%） |
| `best_regressor.pkl` | sklearn RF（设备级86.5%） |
| `numpy_rf_model.pkl` | numpy RF 150树（设备级86.3%） |
| `event_numpy_rf.pkl` | numpy RF 100树（事件级，待完成） |
| `操作总结.md` | 项目综合文档 |
| `progress.md` | 本文件 |

### 4. 已解决的错误

1. **MLlib精度仅52.1%**: maxBins分桶近似导致 → 放弃MLlib，改用numpy手写RF
2. **StandardScaler API不兼容**: Spark 3.5.8不支持 → RF/GBT不需要，移除
3. **`col`变量覆盖**: `for col in [...]`覆盖pyspark的`col()` → 改名为`cat_col`
4. **"could not convert string to float"**: LabelEncoder未应用到分类列 → 提前fit_transform
5. **HDFS事件日志失败**: HDFS已停 → `spark.eventLog.enabled=false`
6. **git commit锁文件**: `.git/index.lock`残留 → `rm -f .git/index.lock`
7. **submit_to_cluster.py超时**: paramiko读取PID超时 → job后台启动后手动检查
8. **Master内存不足(1GB)**: 300树+sklearn依赖超出 → 降为100树+硬编码Top10特征
9. **`task3_event_numpy.py` IndexError**: 两次fit(10特征→13特征)导致特征维度不一致 → 删除重复fit行

### 5. 核心经验

- 设备级预测比事件级稳定得多（聚合消除单次随机性）
- MLlib分桶近似在小数据集精度损失大
- numpy手写RF 71×慢于sklearn（Python解释执行 vs Cython编译），但算法等价
- numpy RF单线程，加CPU核不加速
- Master 1GB内存不够跑13K行×100树事件级RF，需本地执行
- 事件级回归物理上限约72%（单次故障间隔随机性157%）
