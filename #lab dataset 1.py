#lab dataset 1 
#basics:
#clical_mod is my dataset 
#Age at Initial Pathologic Diagnosis is the variable name in the data set 
#age group is a column heading that I have created 
#age group counts stores the number of people within each age group 
#tumor counts counts the number of people with each type of tumor 
import pandas as pd 
import numpy as np 

clinical_mod = pd.read_csv("/Users/anayanath/rainbow/DS-3021-analytics/data/clinical_data_breast_cancer_modified.csv")
print(clinical_mod)

##Question 1 : Is there a relationship between the age at initial pathologic diagnosis 
#and the tumor stage or size in breast cancer patients?

##Pseudocode 1:
#1-group by age groups: 
#2-calculate number of tumor types for each category of tumor (T1,T2 and T3)
#3-within each age group, compute percentage of each tumor type 
#4-write the highest occuring tumor type for each age group 

#understanding the data 
print(clinical_mod.columns)
print(clinical_mod.info())
clinical_mod["Age at Initial Pathologic Diagnosis"] = pd.to_numeric(clinical_mod["Age at Initial Pathologic Diagnosis"])
#age = clinical_mod["Age at Initial Pathologic Diagnosis"]
#print(age)
#print(max(age))
#print(min(age))

#step 1- grouping data into age groups 
#min age is 30 and max age is 88 
#cut function in pandas 
clinical_mod["Age at Initial Pathologic Diagnosis"] = pd.to_numeric(clinical_mod["Age at Initial Pathologic Diagnosis"])

bins = [30, 40, 50, 60, 70, 80, 90]  
labels = ["30-40 years", "40-50 years", "50-60 years", "60-70 years", "70-80 years", "80-90 years"]
clinical_mod["Age Group"] = pd.cut(clinical_mod["Age at Initial Pathologic Diagnosis"], bins=bins, labels=labels)
print(clinical_mod[["Age at Initial Pathologic Diagnosis", "Age Group"]].head())

#counting the number of patients within each age group
age_group_counts = clinical_mod["Age Group"].value_counts()
print(age_group_counts)

## Step 2: 
#calculate number of tumor types for each category of tumor (T1,T2 and T3)
tumor_counts = clinical_mod.groupby("Tumor").size()
print(tumor_counts)


#now creating a table such that person has T1 or T2 or T3 or T4 tumor given that they are in age group 1 
#this is repeated for all 6 age groups
tumor_age_table = clinical_mod.groupby(["Age Group", "Tumor"]).size().unstack()
print(tumor_age_table)

##3-within each age group, compute percentage of each tumor type 
tumor_age_percent = tumor_age_table.div(tumor_age_table.sum(axis=1), axis=0) * 100

# Display the percentage table
print(tumor_age_percent)

#4-write the highest occuring tumor type for each age group 

def calculate_highest_tumor_type(tumor_age_table):
    if not isinstance(tumor_age_table, pd.DataFrame):  # Ensure the input is a DataFrame
        raise ValueError("Input must be a DataFrame. Ensure 'tumor_age_table' is correctly formatted.")

    # Removing rows with all NaN values before performing operations
    tumor_age_table = tumor_age_table.dropna(how="all")

    # using tumor_age_percent from previous code step  
    highest_tumor_type = tumor_age_percent.idxmax(axis=1)  

    return highest_tumor_type
tumor_age_table = clinical_mod.groupby(["Age Group", "Tumor"]).size().unstack(fill_value=0)
highest_tumor_per_age_group = calculate_highest_tumor_type(tumor_age_table)
print(highest_tumor_per_age_group)


### Analysis: It seems like T2 is the most frequently prevalent tumor type 
#in age groups 30-40 years, 40-50 years, 50-60 years, 60-70 years, and 80-90 years
#for age group 70-80 years, the prevalence of tumors T1, T2, T3, and T4 is equally likely 
#this suggests that there is no strong relationship between age at initial pathologic 
#diagnosis and tumor stage, as T2 remains the most prevalent tumor type across most 
#age groups, indicating a consistent pattern regardless of age. The exception in the 
#70-80 years group, where all tumor types are equally likely, may suggest variability
#in tumor progression or detection in older patients, but overall, tumor stage does 
#not appear to be significantly age-dependent.

#background for my exploration in question 2 
#If tumor stage is not correlated with the age at initial pathologic diagnosis, there
#could be other variables that must be correlated with the stage of Tumor developed in the 
#patient. It could be the quality of treatment patients recieve, patients lifestsyle, 
#length of delay between initial diagnosis and a consistent treatment regimen and more. 
#In this question, I want to explore the relationship between Tumor stage and 

##Question  2: Is there an association between tumor stage and ER/PR/HER2-positive or
# negative status.

#After research I learnt that:
#ER-positive means the tumor has estrogen receptors and might grow faster if 
#estrogen is present.
#PR-positive tumors also rely on hormone progesterone to grow.
#HER2 is a protein that helps cells grow. In some cancers, there's too much HER2, 
#making the tumor grow more aggressively.

#pseudocode 2 
#1-seperate ER,PR,HER2 positives and negatives together 
#2-for each stage of tumor, count the number of ER,PR, HER positives 
#3-for each stage of tumor, count the number of ER,PR, HER negatives 
#4-For each stage of tumor, calculate which positive or negative version of the hormone is most 
#frequent

#step 1 
ER_positive = clinical_mod[clinical_mod["ER Status"] == "Positive"]
print(ER_positive)
ER_negative = clinical_mod[clinical_mod["ER Status"] == "Negative"]
print(ER_negative)
PR_positive = clinical_mod[clinical_mod["ER Status"] == "Positive"]
print(PR_positive)
PR_negative = clinical_mod[clinical_mod["ER Status"] == "Negative"]
print(ER_negative)
HER2_positive = clinical_mod[clinical_mod["HER2 Final Status"] == "Positive"]
print(HER2_positive)
HER2_negative = clinical_mod[clinical_mod["HER2 Final Status"] == "Negative"]
print(HER2_negative)

#step 2 
ER_positive_counts = ER_positive.groupby("Tumor")["ER Status"].count()
PR_positive_counts = PR_positive.groupby("Tumor")["PR Status"].count()
HER2_positive_counts = HER2_positive.groupby("Tumor")["HER2 Final Status"].count()
print(ER_positive_counts,PR_positive_counts,HER2_positive_counts)

ER_positive_counts = {"T1": 10, "T2": 39, "T3": 14, "T4": 5}
PR_positive_counts = {"T1": 10, "T2": 39, "T3": 14, "T4": 5}
HER2_positive_counts = {"T1": 4, "T2": 19, "T3": 4, "T4": 0}

# Create a DataFrame
data1 = {
    "Tumor Stage": ["T1", "T2", "T3", "T4"],
    "ER Positive": [ER_positive_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
    "PR Positive": [PR_positive_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
    "HER2 Positive": [HER2_positive_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
}

results_df1 = pd.DataFrame(data1)

# Display the table
print(results_df1)

#step 3 

ER_negative_counts = ER_negative.groupby("Tumor")["ER Status"].count()
PR_negative_counts = PR_negative.groupby("Tumor")["PR Status"].count()
HER2_negative_counts = HER2_negative.groupby("Tumor")["HER2 Final Status"].count()
print(ER_negative_counts,PR_negative_counts,HER2_negative_counts)

ER_positive_counts = {"T1": 5, "T2": 25, "T3": 5, "T4":1}
PR_positive_counts = {"T1": 5, "T2": 25, "T3": 5, "T4": 1}
HER2_positive_counts = {"T1": 11, "T2": 46, "T3": 14, "T4": 6}

data2 = {
    "Tumor Stage": ["T1", "T2", "T3", "T4"],
    "ER Negative": [ER_negative_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
    "PR Negative": [PR_negative_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
    "HER2 Negative": [HER2_negative_counts.get(stage, 0) for stage in ["T1", "T2", "T3", "T4"]],
}

results_df2 = pd.DataFrame(data2)

# Display the table
print(results_df2)

#step 4
def compute_most_frequent_hormone(results_df1):
    results_df1 = results_df1.copy()  #ensuring the original dataframe is unmodified 
    results_df1["Most Frequent Hormone"] = results_df1[["ER Positive", "PR Positive", "HER2 Positive"]].apply(
        lambda row: " & ".join(row.index[row == row.max()]), axis=1
    )
    return results_df1

# Running the function using results_df1
final_results_df1 = compute_most_frequent_hormone(results_df1)
# Displaying results
print(final_results_df1)




#Analysis- The results show that the most frequent hormone receptors  for all 4 types of Tumors are ER and PR 
#positive. This suggests that that ER-positive and PR-positive statuses are dominant across all tumor 
#stages, indicating that most breast cancer tumors rely on hormone receptors for growth regardless of stage. 
#Since HER2 positivity is less frequent, tumor stage does not appear to be strongly associated with ER/PR/HER2 
#status, but hormone receptor-positive tumors may be more common overall.







