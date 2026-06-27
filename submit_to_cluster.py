"""
提交Spark任务到集群
用法: python submit_to_cluster.py <脚本名>
例:  python submit_to_cluster.py task3_spark_final.py
"""
import paramiko, sys, os, time

# ===== 集群配置 =====
MASTER_IP = "192.168.149.128"
SSH_USER  = "root"
SSH_PASS  = "123456zz"
SPARK_CMD = "/root/software/spark-3.5.8/bin/spark-submit"
SPARK_OPTS = "--master spark://Master001:7077 --conf spark.eventLog.enabled=false --driver-memory 512m --executor-memory 512m --total-executor-cores 12"

# ===== 1. 上传脚本 =====
script = sys.argv[1] if len(sys.argv) > 1 else "task3_spark_final.py"
print(f"上传 {script} 到 Master...")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(MASTER_IP, username=SSH_USER, password=SSH_PASS, timeout=15)

sftp = c.open_sftp()
sftp.put(script, f"/root/{script}")
sftp.close()
print("上传完成")

# ===== 2. 提交任务 =====
print(f"提交Spark任务: {script}")
c.exec_command("pkill -9 -f spark-submit 2>/dev/null; sleep 2", timeout=10)

stdin, stdout, stderr = c.exec_command(
    f"cd /root && nohup {SPARK_CMD} {SPARK_OPTS} {script} > task3_output.txt 2>&1 & echo PID=$!",
    timeout=15)
print("PID:", stdout.read().decode().strip()[-50:])

# ===== 3. 等待并获取结果 =====
print("等待任务完成 (每30秒检查一次)...")
while True:
    time.sleep(30)
    try:
        sftp = c.open_sftp()
        sftp.get("/root/task3_output.txt", "cluster_output.txt")
        sftp.close()

        with open("cluster_output.txt", "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if "Successfully stopped" in content or "EXIT_CODE" in content:
            print("\n任务完成! 结果:\n" + "=" * 60)
            for line in content.split("\n"):
                if any(k in line for k in ["MAE", "MAPE", "1-MAPE", "最优", "精度", "达标", "加分"]):
                    print(line.strip()[-250:])
            break
        else:
            lines = content.count("\n")
            print(f"  运行中... ({lines}行日志)", end="\r")
    except:
        print("  等待Spark初始化...", end="\r")

c.close()
