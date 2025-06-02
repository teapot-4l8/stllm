# import pickle as pkl
# import pandas as pd
# file_path = "./bike_drop/adj_mx.pkl"
# with open(file_path, "rb") as f:
#     object = pkl.load(f)
    
# df = pd.DataFrame(object)
# df.to_csv(r'bike_drop/adj_mx.csv')

import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"data\evdata\adj_filter.csv")  # 可添加参数如encoding='utf-8', sep=','等

# 保存为PKL文件
df.to_pickle("adj_mx.pkl")  # 默认使用最高效的protocol

# 验证：重新加载PKL文件
loaded_df = pd.read_pickle("adj_mx.pkl")
print(loaded_df.head())