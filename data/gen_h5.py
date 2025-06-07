"""
把深圳1小时间隔数据集的occupancy转成h5格式
"""

import pandas as pd

# 1. 读取 CSV 文件（自动识别索引或设置索引列）
df = pd.read_csv('./volume/volume.csv', index_col=0)  # index_col=0 表示将第一列作为索引

# 2. 保存为 HDF5 文件（默认保存索引）
df.to_hdf('./h5data/volume.h5', key='data', mode='w')

# # 索引也会被一起加载回来。
# df_loaded = pd.read_hdf('data\evdata\occupancy.h5', key='data')
# df_loaded.index = pd.to_datetime(df.index, format='%m/%d/%Y %H:%M')


# --------------------------------------------------------
# 生成price.csv
# e_prc = pd.read_csv('./evdata/e_price.csv', index_col=0, header=0)  # electricity price
# s_prc = pd.read_csv('./evdata/s_price.csv', index_col=0, header=0)  # service price

# prc = s_prc + e_prc
# # df.to_hdf('data\evdata\price.h5', key='data', mode='w')
# # df_loaded = pd.read_hdf('data\evdata\price.h5', key='data')
# prc.to_csv("./evdata/price.csv")
