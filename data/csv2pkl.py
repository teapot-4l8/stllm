# import pickle as pkl
# import pandas as pd
# file_path = "./bike_drop/adj_mx.pkl"
# with open(file_path, "rb") as f:
#     object = pkl.load(f)
    
# df = pd.DataFrame(object)
# df.to_csv(r'bike_drop/adj_mx.csv')

import pandas as pd
import pickle

# Adjust the path to your CSV and PKL files
csv_path = r"data/evdata/adj_filter.csv"
pkl_path = r"data/evdata/adj_mx.pkl"

# Read CSV, tell pandas the first column is index
df = pd.read_csv(csv_path, index_col=0)

# Get the numpy array (without index and header)
adj_array = df.values

# Save as pickle file
with open(pkl_path, "wb") as f:
    pickle.dump(adj_array, f)