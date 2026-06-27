"""
任务1 Spark版：数据清洗及数据集构建
运行方式: spark-submit task1_spark.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, when, trim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "file:///root/设备预测数据.xlsx"  # Spark会通过pandas读取Excel
OUTPUT_FILE = "file:///root/cleaned_afc_data_spark.csv"

# 列名映射
COLUMN_MAPPING = {
    'assetnum': '设备编号', 'wonum': '工单号', 'location': '位置编码',
    'station_name': '车站名称', 'cust_linenum': '线路编号',
    'current_faildate': '故障时间', 'prev_faildate': '上次故障时间',
    'total_failure_count': '累计故障次数', 'is_first_failure': '是否首次故障',
    'worktype': '工单类型', 'description': '故障描述',
    'cust_brand': '设备品牌', 'cust_subsys': '子系统',
    'failurecode': '故障代码', 'problemcode': '问题代码',
    'failurecode_name': '故障代码名称', 'problemcode_name': '问题代码名称',
    'repair_start_time': '维修开始时间', 'repair_end_time': '维修完成时间',
    'repair_type': '维修类型', 'repair_location': '维修位置',
    'repair_duration_hours': '维修时长_小时', 'start_of_operation': '投运日期',
    'cust_fixby': '维修人员编号', 'responsible_person': '负责人',
    'problem_desc': '问题详细描述', 'cause_desc': '原因描述',
    'remedy_desc': '解决方法描述', 'manual_remark': '手动备注',
}

CORE_COLS = ['设备编号', '故障时间', '维修完成时间']
TIME_COLS = ['故障时间', '上次故障时间', '维修开始时间', '维修完成时间', '投运日期']
FILL_VALS = {'设备品牌': '未知', '子系统': '未知', '故障代码': '未知', '问题代码': '未知',
             '故障代码名称': '未知', '问题代码名称': '未知', '维修类型': '未知',
             '问题详细描述': '无', '原因描述': '无', '解决方法描述': '无', '手动备注': '无'}

if __name__ == '__main__':
    spark = SparkSession.builder.appName("Task1-DataCleaning").getOrCreate()

    # 用pandas读取Excel（Spark不直接支持Excel）
    import pandas as pd
    logger.info("读取Excel数据...")
    pdf = pd.read_excel('/root/设备预测数据.xlsx')
    df = spark.createDataFrame(pdf)
    logger.info(f"原始数据: {df.count()} 行, {len(df.columns)} 列")

    # 1. 列名标准化
    for old, new in COLUMN_MAPPING.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # 2. 去重
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    logger.info(f"去重: {before} -> {after} (删除{before - after}条)")

    # 3. 删除核心字段缺失
    for c in CORE_COLS:
        if c in df.columns:
            df = df.filter(col(c).isNotNull() & (trim(col(c)) != ''))
    logger.info(f"核心字段过滤后: {df.count()} 行")

    # 4. 填充缺失值
    for c, v in FILL_VALS.items():
        if c in df.columns:
            df = df.fillna({c: v})

    # 5. 时间字段解析
    for c in TIME_COLS:
        if c in df.columns:
            df = df.withColumn(c, to_timestamp(col(c), 'yyyy-MM-dd HH:mm:ss'))

    # 6. 时间逻辑校验
    df = df.filter(col('维修完成时间') >= col('故障时间'))
    df = df.filter(col('故障时间') >= '2020-01-01')
    df = df.filter(col('故障时间') <= '2026-12-31')

    final_count = df.count()
    logger.info(f"清洗完成: {final_count} 行")

    # 7. 输出
    pdf_out = df.toPandas()
    pdf_out.to_csv('/root/cleaned_afc_data_spark.csv', index=False, encoding='utf-8-sig')
    logger.info(f"已保存: /root/cleaned_afc_data_spark.csv")

    spark.stop()
