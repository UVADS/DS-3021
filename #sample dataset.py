#sample dataset 
import pandas as pd

# Provide the full path to the file
file_path = '/Users/anayanath/Downloads/ClassData.csv'  

# Read the CSV file
data = pd.read_csv(file_path)

# Print the first few rows of the dataset
print(data.head())