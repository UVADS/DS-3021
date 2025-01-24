#studentdata.csv
import pandas as pd  # Import pandas library

# Read the CSV file into a DataFrame
student_data = pd.read_csv('studentdata.csv',encoding='latin1')

# Display the first few rows of the data
print(student_data.head())
student_data
