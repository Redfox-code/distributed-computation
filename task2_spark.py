"""
任务2 Spark版：数据加工及特征筛选
运行方式: spark-submit task2_spark.py
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, to_timestamp, hour, dayofweek, month, quarter, when,
    datediff, lag, mean as spark_mean, stddev, count, first, last,
    collect_list, size, coalesce, lit
)
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Task2-FeatureEngineering") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # Driver端用pandas加载CSV，转为Spark DataFrame
    logger.info("Driver端加载清洗后数据...")
    import pandas as pd
    pdf_raw = pd.read_csv('/root/cleaned_afc_data.csv', encoding='utf-8-sig')
    # 时间列保持字符串，Spark解析
    df = spark.createDataFrame(pdf_raw)
    # Spark解析时间列
    for c in ['故障时间', '上次故障时间', '维修开始时间', '维修完成时间', '投运日期']:
        df = df.withColumn(c, to_timestamp(col(c), 'yyyy-MM-dd HH:mm:ss'))

    # 按设备排序
    df = df.orderBy('设备编号', '故障时间')

    # 过滤只有1条记录的设备
    dev_counts = df.groupBy('设备编号').count().filter(col('count') >= 2)
    df = df.join(dev_counts.select('设备编号'), on='设备编号', how='inner')
    logger.info(f"过滤后(>=2条): {df.count()} 行")

    # ===== 使用窗口函数计算特征 =====
    win = Window.partitionBy('设备编号').orderBy('故障时间')

    # 故障间隔（目标变量）
    df = df.withColumn('故障间隔_小时',
        (col('故障时间').cast('long') - lag('故障时间', 1).over(win).cast('long')) / 3600.0)

    # 删除首条（间隔=NaN）
    df = df.filter(col('故障间隔_小时').isNotNull())
    df = df.filter(col('故障间隔_小时') > 0)

    # 时间特征
    df = df.withColumn('故障小时', hour('故障时间')) \
           .withColumn('故障星期', dayofweek('故障时间') - 1) \
           .withColumn('故障月份', month('故障时间')) \
           .withColumn('是否周末', when(dayofweek('故障时间').isin([1, 7]), 1).otherwise(0))

    # 维修特征
    df = df.withColumn('维修响应_小时',
        (col('维修开始时间').cast('long') - col('故障时间').cast('long')) / 3600.0)
    df = df.withColumn('维修响应_小时', when(col('维修响应_小时') < 0, 0).otherwise(col('维修响应_小时')))
    df = df.withColumn('维修类型_编码', when(col('维修类型') == 'CBM', 1).otherwise(0))

    logger.info(f"有效样本: {df.count()} 行")

    # 转为Pandas做后续处理（数据量小，适合单机）
    pdf = df.toPandas()

    # 分类编码
    for col_name in ['设备品牌', '子系统', '故障代码名称']:
        pdf[col_name + '_编码'] = LabelEncoder().fit_transform(pdf[col_name].astype(str))
    for col_name in ['问题代码名称', '线路编号']:
        freq = pdf[col_name].value_counts(normalize=True)
        pdf[col_name + '_频次'] = pdf[col_name].map(freq)

    # 设备内历史统计特征（无泄漏）
    result_rows = []
    for dev_id, dev_df in pdf.groupby('设备编号'):
        dev_df = dev_df.sort_values('故障时间').reset_index(drop=True)
        intervals = dev_df['故障间隔_小时'].values
        repair_durs = dev_df['维修时长_小时'].values
        hist_ints, hist_durs = [], []

        for i in range(len(dev_df)):
            nh = len(hist_ints)
            havg = np.mean(hist_ints) if hist_ints else intervals[i]
            hlast = hist_ints[-1] if hist_ints else intervals[i]
            r3 = np.mean(hist_ints[-3:]) if nh >= 3 else (np.mean(hist_ints) if hist_ints else intervals[i])
            rdur = np.mean(hist_durs) if hist_durs else repair_durs[0]
            ldur = hist_durs[-1] if hist_durs else repair_durs[0]
            days = max((dev_df['故障时间'].iloc[i] - dev_df['故障时间'].iloc[0]).total_seconds() / 86400, 1)

            row = dev_df.iloc[i].to_dict()
            row.update({
                '历史故障次数': nh, '历史平均间隔': havg, '上次间隔': hlast,
                '最近3次平均间隔': r3, '历史平均维修时长': rdur, '上次维修时长': ldur,
                '设备运行天数': days, '历史故障频率': nh / days,
            })
            result_rows.append(row)
            if not np.isnan(intervals[i]): hist_ints.append(intervals[i])
            hist_durs.append(repair_durs[i])

    result_df = pd.DataFrame(result_rows)

    # 特征筛选
    feature_cols = [
        '故障小时', '故障星期', '故障月份', '是否周末',
        '维修时长_小时', '维修响应_小时', '维修类型_编码',
        '历史故障次数', '历史平均间隔', '上次间隔', '最近3次平均间隔',
        '历史平均维修时长', '上次维修时长', '设备运行天数', '历史故障频率',
        '设备品牌_编码', '子系统_编码', '故障代码名称_编码',
        '问题代码名称_频次', '线路编号_频次',
    ]
    target_col = '故障间隔_小时'

    # 过滤异常值
    result_df = result_df[result_df[target_col] >= 10]
    upper = result_df[target_col].quantile(0.92)
    result_df = result_df[result_df[target_col] <= upper]

    # 保留特征+目标
    keep_cols = ['设备编号'] + feature_cols + [target_col]
    result_df = result_df[keep_cols].dropna()

    result_df.to_csv('/root/modeling_dataset_spark.csv', index=False, encoding='utf-8-sig')
    logger.info(f"建模数据集: {len(result_df)} 行 x {len(keep_cols)} 列")
    logger.info(f"已保存: /root/modeling_dataset_spark.csv")

    spark.stop()
