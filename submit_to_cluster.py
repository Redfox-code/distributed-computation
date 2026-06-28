"""
提交任务到Spark集群 + 下载结果
用法:
  python submit_to_cluster.py <脚本名>
  python submit_to_cluster.py <脚本名> --driver 1g
  python submit_to_cluster.py <脚本名> --no-spark

例:
  python submit_to_cluster.py task2_event_spark.py
  python submit_to_cluster.py task3_event_spark_full.py --driver 1g
  python submit_to_cluster.py task3_event_numpy.py --no-spark
"""
import paramiko, sys, os, time

MASTER_IP   = "192.168.149.128"
SSH_USER    = "root"
SSH_PASS    = "123456zz"
SPARK_HOME  = "/root/software/spark-3.5.8"
PYTHON_BIN  = "/root/anaconda3/envs/pyspark/bin/python"
LOG_FILE    = "/root/task3_output.txt"
SPARK_MASTER = "spark://Master001:7077"
DRIVER_MEM   = "1g"
EXECUTOR_MEM = "512m"
EXECUTOR_CORES = 12

# ---- 解析参数 ----
args = sys.argv[1:]
if not args:
    print("用法: python submit_to_cluster.py <脚本名> [--driver 1g] [--no-spark]")
    sys.exit(1)

script = args[0]; use_spark = True
i = 1
while i < len(args):
    if args[i] == '--driver' and i+1 < len(args): DRIVER_MEM = args[i+1]; i += 2
    elif args[i] == '--no-spark': use_spark = False; i += 1
    else: i += 1

if not os.path.exists(script): print(f"错误: 文件不存在 - {script}"); sys.exit(1)

# ---- 1. 连接 ----
print(f"连接 Master ({MASTER_IP})...")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(MASTER_IP, username=SSH_USER, password=SSH_PASS, timeout=15)

# ---- 2. 上传 ----
print(f"上传 {script} -> /root/{script}")
sftp = c.open_sftp()
sftp.put(script, f"/root/{script}")
for df in ['cleaned_afc_data.csv', 'modeling_dataset.csv']:
    if os.path.exists(df): sftp.put(df, f"/root/{df}"); print(f"上传 {df} -> /root/{df}")
sftp.close()
print("上传完成\n")

# ---- 3. 集群状态 ----
if use_spark:
    stdin, stdout, stderr = c.exec_command(
        'curl -s http://Master001:8080 2>&1 | grep -oE "Alive Workers:.*<"', timeout=10)
    alive = stdout.read().decode().strip()
    if 'Alive Workers' in alive: print(f"集群: {alive.replace('<','').replace('/strong>','')}")
    else: print("警告: 无法获取集群状态")

# ---- 4. 清理 ----
print("清理旧进程...")
c.exec_command('pkill -9 -f spark-submit 2>/dev/null', timeout=10); time.sleep(3)
stdin, stdout, stderr = c.exec_command('ps aux | grep -v grep | grep -c spark-submit', timeout=10)
if stdout.read().decode().strip() != '0':
    c.exec_command('pkill -9 -f spark-submit 2>/dev/null', timeout=10); time.sleep(2)
c.exec_command(f'rm -f {LOG_FILE}', timeout=5)
print("清理完成\n")

# ---- 5. 提交 ----
if use_spark:
    opts = f"--master {SPARK_MASTER} --conf spark.eventLog.enabled=false --driver-memory {DRIVER_MEM} --executor-memory {EXECUTOR_MEM} --total-executor-cores {EXECUTOR_CORES}"
    cmd = f"cd /root && nohup {SPARK_HOME}/bin/spark-submit {opts} {script} > {LOG_FILE} 2>&1 &"
    print(f"提交: spark-submit (driver={DRIVER_MEM})")
else:
    cmd = f"cd /root && nohup {PYTHON_BIN} {script} > {LOG_FILE} 2>&1 &"
    print(f"提交: 纯Python")

c.exec_command(cmd, timeout=10); time.sleep(5)

stdin, stdout, stderr = c.exec_command('ps aux | grep -v grep | grep -E "spark-submit|task2_event|task3_event" | head -3', timeout=10)
ps_out = stdout.read().decode().strip()
print(f"{'OK 任务已启动' if ps_out else '警告: 进程未出现'}")

# ---- 6. 监控 ----
print(f"\n{'='*50}")
print(f"监控中 (Ctrl+C 停止监控, 任务继续后台运行)")
print(f"手动: ssh root@{MASTER_IP} 'tail -f {LOG_FILE}'")
print(f"{'='*50}")

last_lines = 0; check_count = 0
try:
    while True:
        time.sleep(15); check_count += 1
        try:
            sftp = c.open_sftp()
            sftp.get(LOG_FILE, "cluster_output.txt"); sftp.close()
            with open("cluster_output.txt", "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = content.split('\n'); total_lines = len(lines)

            is_done = any(k in content for k in ['完成!', 'ShutdownHook', 'Successfully stopped'])

            if is_done:
                print(f"\n{'='*55}")
                print(" 任务完成! 结果:")
                print("="*55)
                for line in lines:
                    if any(k in line for k in ['MAE','RMSE','MAPE','1-MAPE','R2','R²','Top5',
                                                '完成','Scout','主RF','Stage','分布式','集群',
                                                '特征','验证','保留','1-MAPE']):
                        if 'INFO' in line:
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3: print(f"  {parts[-1][:150]}")
                        else: print(f"  {line.strip()[:150]}")

                # ---- 下载结果文件 ----
                print(f"\n{'='*55}")
                print(" 下载结果文件...")
                sftp = c.open_sftp()
                # 图表数据 + 建模数据
                for rf in ['chart_data.json', 'modeling_dataset_event.csv']:
                    try:
                        sftp.get(f'/root/{rf}', rf)
                        print(f"  OK {rf}")
                    except: pass
                sftp.close()

                # 本地生成图表 (有中文字体)
                if os.path.exists('chart_data.json') and os.path.exists('plot_task2.py'):
                    print("\n 生成本地图表...")
                    import subprocess
                    subprocess.run([sys.executable, 'plot_task2.py'])
                break

            # 进度显示
            if total_lines > last_lines:
                for line in lines[last_lines:total_lines]:
                    if any(k in line for k in ['Step','Stage','Phase','Top10','Scout',
                                                '树 ','RF完成','事件数','训练:','广播',
                                                '分区','特征工程','过滤后']):
                        if 'INFO' in line:
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3: print(f"  [{check_count*15}s] {parts[-1][:150]}")
                last_lines = total_lines
            if check_count % 8 == 0:
                print(f"  ...运行中 ({check_count*15//60}分钟, {total_lines}行)...")

        except Exception as e:
            if check_count <= 2: print(f"  等待初始化...")
except KeyboardInterrupt:
    print(f"\n监控停止 (任务在后台继续)")
    print(f"查看: ssh root@{MASTER_IP} 'tail -f {LOG_FILE}'")

c.close()
