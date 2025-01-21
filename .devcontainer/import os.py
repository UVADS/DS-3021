import os

file_path = r'/workspaces/DS-3021/alcohol-available-for-consumption-year-ended-december-2023.csv'
print(os.path.exists(file_path))  # Should return True if the file exists
