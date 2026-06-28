"""
提交任务到Spark集群
用法:
  python submit_to_cluster.py <脚本名>
  python submit_to_cluster.py <脚本名> --driver 1g
  python submit_to_cluster.py <脚本名> --no-spark          # 纯Python脚本, 不用spark-submit

例:
  python submit_to_cluster.py task3_spark_final.py
  python submit_to_cluster.py task3_event_spark_distributed.py
  python submit_to_cluster.py task3_event_numpy.py --no-spark
  python submit_to_cluster.py task3_event_spark_full.py --driver 1g
"""
import paramiko, sys, os, time

# ===== 集群配置 =====
MASTER_IP   = "192.168.149.128"
SSH_USER    = "root"
SSH_PASS    = "123456zz"
SPARK_HOME  = "/root/software/spark-3.5.8"
PYTHON_BIN  = "/root/anaconda3/envs/pyspark/bin/python"
LOG_FILE    = "/root/task3_output.txt"

# 默认Spark参数
SPARK_MASTER = "spark://Master001:7077"
DRIVER_MEM   = "1g"
EXECUTOR_MEM = "512m"
EXECUTOR_CORES = 12

# ===== 解析参数 =====
args = sys.argv[1:]
if not args:
    print("用法: python submit_to_cluster.py <脚本名> [--driver 1g] [--no-spark]")
    print("示例:")
    print("  python submit_to_cluster.py task3_event_spark_full.py")
    print("  python submit_to_cluster.py task3_event_numpy.py --no-spark")
    sys.exit(1)

script = args[0]
use_spark = True  # 默认用spark-submit

i = 1
while i < len(args):
    if args[i] == '--driver' and i + 1 < len(args):
        DRIVER_MEM = args[i + 1]; i += 2
    elif args[i] == '--no-spark':
        use_spark = False; i += 1
    else:
        i += 1

if not os.path.exists(script):
    print(f"错误: 文件不存在 - {script}")
    sys.exit(1)

# ===== 1. 连接集群 =====
print(f"连接 Master ({MASTER_IP})...")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect(MASTER_IP, username=SSH_USER, password=SSH_PASS, timeout=15)
except Exception as e:
    print(f"连接失败: {e}")
    sys.exit(1)

# ===== 2. 上传文件 =====
print(f"上传 {script} → /root/{script}")
sftp = c.open_sftp()
sftp.put(script, f"/root/{script}")

# 如果数据文件存在也上传 (确保最新)
for data_file in ['cleaned_afc_data.csv', 'modeling_dataset.csv']:
    if os.path.exists(data_file):
        sftp.put(data_file, f"/root/{data_file}")
        print(f"上传 {data_file} → /root/{data_file}")
sftp.close()
print("上传完成\n")

# ===== 3. 检查集群状态 (仅spark模式) =====
if use_spark:
    stdin, stdout, stderr = c.exec_command(
        'curl -s http://Master001:8080 2>&1 | grep -oE "Alive Workers:.*<"', timeout=10)
    alive = stdout.read().decode().strip()
    if 'Alive Workers' in alive:
        print(f"集群状态: {alive.replace('<','').replace('/strong>','').replace('</li','')}")
    else:
        print("⚠ 警告: 无法获取集群状态, 尝试继续...")

# ===== 4. 杀旧进程 =====
print("清理旧进程...")
# 先杀掉旧的spark-submit (如果有)
c.exec_command('pkill -9 -f spark-submit 2>/dev/null; sleep 1', timeout=10)
# 等进程彻底退出
time.sleep(3)

# 确认已清理
stdin, stdout, stderr = c.exec_command('ps aux | grep -v grep | grep -c spark-submit', timeout=10)
remaining = stdout.read().decode().strip()
if remaining != '0':
    print(f"  仍有 {remaining} 个旧进程, 强制清理...")
    c.exec_command('pkill -9 -f spark-submit 2>/dev/null; sleep 2', timeout=10)

# 清空旧日志
c.exec_command(f'rm -f {LOG_FILE}', timeout=5)
print("清理完成\n")

# ===== 5. 提交任务 =====
if use_spark:
    spark_opts = (f"--master {SPARK_MASTER} "
                  f"--conf spark.eventLog.enabled=false "
                  f"--driver-memory {DRIVER_MEM} "
                  f"--executor-memory {EXECUTOR_MEM} "
                  f"--total-executor-cores {EXECUTOR_CORES}")
    submit_cmd = (f"cd /root && nohup {SPARK_HOME}/bin/spark-submit {spark_opts} "
                  f"{script} > {LOG_FILE} 2>&1 &")
    print(f"提交模式: spark-submit (driver={DRIVER_MEM})")
else:
    submit_cmd = (f"cd /root && nohup {PYTHON_BIN} {script} "
                  f"> {LOG_FILE} 2>&1 &")
    print(f"提交模式: 纯Python (不启动JVM, 省内存)")

c.exec_command(submit_cmd, timeout=10)
time.sleep(5)

# 确认启动
stdin, stdout, stderr = c.exec_command('ps aux | grep -v grep | grep -E "spark-submit|task3_event" | head -3', timeout=10)
ps_out = stdout.read().decode().strip()
if ps_out:
    print(f"✓ 任务已启动:\n  {ps_out[:200]}")
else:
    print("⚠ 进程未出现, 查看初始日志...")
    stdin, stdout, stderr = c.exec_command(f'head -10 {LOG_FILE} 2>/dev/null', timeout=10)
    early = stdout.read().decode().strip()
    if early:
        print(f"  日志: {early[:300]}")
    else:
        print("  日志为空, 可能启动失败")

# ===== 6. 监控进度 =====
print("\n" + "=" * 50)
print("监控训练进度 (每15秒检查, Ctrl+C 退出监控)")
print(f"手动查看: ssh root@{MASTER_IP} 'tail -f {LOG_FILE}'")
print("=" * 50)

last_lines = 0
check_count = 0

try:
    while True:
        time.sleep(15)
        check_count += 1

        try:
            sftp = c.open_sftp()
            sftp.get(LOG_FILE, "cluster_output.txt")
            sftp.close()

            with open("cluster_output.txt", "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            lines = content.split('\n')
            total_lines = len(lines)

            # 检测是否完成
            is_done = any(kw in content for kw in
                         ['完成!', 'ShutdownHook', 'Successfully stopped',
                          'SparkContext: Successfully stopped'])

            if is_done:
                # 打印结果
                print("\n" + "=" * 55)
                print(" 任务完成! 结果:")
                print("=" * 55)
                for line in lines:
                    if any(k in line for k in ['MAE', 'RMSE', 'MAPE', '1-MAPE',
                                                'R²', 'R2', '最优', '★★', '★', '✓',
                                                '完成', 'Scout', '主RF', 'Stage',
                                                '分布式', '集群', '总分', '加速比']):
                        # 只打印中文部分, 跳过Spark INFO行
                        if 'INFO' in line:
                            # 提取INFO之后的内容
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3:
                                print(f"  {parts[-1][:150]}")
                        else:
                            print(f"  {line.strip()[:150]}")
                break

            # 显示新日志行 (训练进度)
            if total_lines > last_lines:
                new_lines = lines[last_lines:total_lines]
                for line in new_lines:
                    if any(k in line for k in ['Step', 'Stage', 'Phase',
                                                '特征重要性', 'Top10', 'Scout',
                                                '树 ', 'RF完成', '事件数',
                                                '训练:', '广播', '分区']):
                        if 'INFO' in line:
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3:
                                print(f"  [{check_count*15}s] {parts[-1][:150]}")
                        else:
                            print(f"  [{check_count*15}s] {line.strip()[:150]}")
                last_lines = total_lines

            # 每2分钟显示简要状态
            if check_count % 8 == 0:
                print(f"  ...运行中 ({check_count*15//60}分钟, {total_lines}行日志)...")

        except Exception as e:
            if check_count <= 2:
                print(f"  等待Spark初始化... ({e})")
            else:
                # 可能是连接断开, 尝试重连
                try:
                    c.close()
                    c.connect(MASTER_IP, username=SSH_USER, password=SSH_PASS, timeout=15)
                    print("  连接已恢复")
                except:
                    pass

except KeyboardInterrupt:
    print("\n\n监控已停止 (任务仍在集群后台运行)")
    print(f"查看进度: ssh root@{MASTER_IP} 'tail -f {LOG_FILE}'")
    print(f"下载日志: scp root@{MASTER_IP}:{LOG_FILE} ./cluster_output.txt")

c.close()
