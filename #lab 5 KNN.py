#lab 5 KNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

#Q1:
#Research question from previous lab:
#working to develop a model that can predict students' graduate rates (grad_100_value) 
#target variable:
#grad_100_value
college_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/cc_institution_details.csv")
#print(college_data["grad_100_value"].head())
#the target variable is a continuous numeric variable (float). Since kNN works for classification tasks, I 
#need to convert it into categorical classes.
#Low Graduation Rate (0 - 33%) → Class 0
#Medium Graduation Rate (33 - 66%) → Class 1
#High Graduation Rate (66 - 100%) → Class 2
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
sns.displot(
    data=college_data.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
#plt.show()

# Drop columns with more than 50% missing values
threshold = len(college_data) * 0.5
college_data = college_data.dropna(thresh=threshold, axis=1)

# Fill numerical missing values with median
num_cols = college_data.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    college_data[col].fillna(college_data[col].median(), inplace=True)

# Fill categorical missing values with mode
cat_cols = college_data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    college_data[col].fillna(college_data[col].mode()[0], inplace=True)
