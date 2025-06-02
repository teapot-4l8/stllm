import h5py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
# 1. Set the path to your H5 file
h5_file_path = r"D:\_________________________PythonProject\ST-LLM-Plus-main\data\h5data\nyc-bike.h5" 

# 2. Set the folder where you want to save the CSV files
output_folder = "csv_data"
# --- End of Configuration ---


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

try:
    # Open the H5 file in read mode
    with h5py.File(h5_file_path, 'r') as h5f:
        # List all the datasets available in the H5 file
        print(f"Datasets found in '{os.path.basename(h5_file_path)}': {list(h5f.keys())}")

        # Loop through the keys you want to convert
        # Your original script mentions 'bike_pick' and 'bike_drop'
        for key in ['bike_pick', 'bike_drop']:
            if key in h5f:
                print(f"Processing dataset: '{key}'...")
                
                # Extract the dataset as a NumPy array
                data = h5f[key][:] # The [:] loads the full dataset into memory
                
                # Convert the NumPy array to a Pandas DataFrame
                df = pd.DataFrame(data)
                
                # Define the output CSV file path
                output_csv_path = os.path.join(output_folder, f"{key}.csv")
                
                # Save the DataFrame to a CSV file
                # index=False prevents pandas from writing row indices into the file
                df.to_csv(output_csv_path, index=False)
                
                print(f"Successfully converted '{key}' to '{output_csv_path}'")
                print("-" * 20)
            else:
                print(f"Warning: Dataset '{key}' not found in the H5 file.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{h5_file_path}'")
except Exception as e:
    print(f"An error occurred: {e}")

