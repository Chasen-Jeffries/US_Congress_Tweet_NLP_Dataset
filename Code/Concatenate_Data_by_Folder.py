import pandas as pd
import glob
import os

def concatenate_csv_files(directory_path):
    # Get the name of the directory to use as the output file name
    folder_name = os.path.basename(directory_path)
    output_file = f"{folder_name}_concatenated.csv"
    
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(f"{directory_path}/*.csv")
    # Initialize a list to store individual DataFrames
    dfs = []
    
    # Loop through each CSV file and append its DataFrame to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)

# Usage
directory_path = './2023'  # Replace with your directory path
concatenate_csv_files(directory_path)
print('Concatenation Complete')