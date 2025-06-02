# import pandas as pd

# # 1. 读取 CSV 文件（自动识别索引或设置索引列）
# df = pd.read_csv('data\evdata\occupancy.csv', index_col=0)  # index_col=0 表示将第一列作为索引

# # 2. 保存为 HDF5 文件（默认保存索引）
# df.to_hdf('data\evdata\occupancy.h5', key='data', mode='w')

# # 索引也会被一起加载回来。
# df_loaded = pd.read_hdf('data\evdata\occupancy.h5', key='data')
# df_loaded.index = pd.to_datetime(df.index, format='%m/%d/%Y %H:%M')


import numpy as np

# Example: your array (replace with your real array)
x = np.random.rand(5, 4, 3, 2)  # Replace with your actual array

# Show all numbers regardless of array size
np.set_printoptions(threshold=np.inf)

print(x)

np.savetxt("my_array.txt", x.reshape(-1, x.shape[-1]))
