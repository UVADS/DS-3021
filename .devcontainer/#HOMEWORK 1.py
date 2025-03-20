#Lab 5- KNN
import pandas as pd
import numpy as np

# Load dataset
college_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/cc_institution_details.csv")

# Convert graduation rate into categories
def categorize_grad_rate(value):
    if value < 33:
        return 0  # Low graduation rate
    elif 33 <= value < 66:
        return 1  # Medium graduation rate
    else:
        return 2  # High graduation rate

college_data["grad_category"] = college_data["grad_100_value"].apply(categorize_grad_rate)
college_data["grad_category"] = college_data["grad_category"].astype("category")

# Drop the original continuous variable since we have converted it
college_data.drop("grad_100_value", axis=1, inplace=True)