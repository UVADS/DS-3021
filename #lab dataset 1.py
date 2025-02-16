#lab dataset 1 
#basics:
#clinical_mod is my dataset 
#Age at Initial Pathologic Diagnosis is the variable name in the data set 
#age group is a column heading that I have created 
#age group counts stores the number of people within each age group 
#tumor counts counts the number of people with each type of tumor 
import pandas as pd 
import numpy as np 

clinical_mod = pd.read_csv("/workspaces/DS-3021-analytics-1/data/clinical_data_breast_cancer_modified.csv")
print(clinical_mod)


##Question 1 : Is there a relationship between the age at initial pathologic diagnosis 
#and the tumor stage or size in breast cancer patients?
#question 1 type: Correlation 

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
#Create a tumor distribution table: count occurrences of each tumor type within each age group
tumor_age_table = clinical_mod.groupby(["Age Group", "Tumor"]).size().unstack(fill_value=0)
#checking the function to determine the highest occurring tumor type per age group
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


#Question 2 - redo 
#Question 2 type: Group Difference Analysis 
#Are there differences in the average overall survival time (OS Time) of 
#patients based on AJCC stage? 
#question type: comparative or group difference analysis 

# Pseudocode:
#1-group by AJCC stage 
#2-calculate average overall survival time for each stage
#3-calculate difference between average survival time between AJCC stages 
#4-visually comparing average survival times across AJCC Stages 

#step 1 group by AJCC stage 
print(clinical_mod.info()) #this reveals the different data types of columns
#changing AJCC stage to categorical variable
#clinical_mod['AJCC Stage'] = clinical_mod['AJCC Stage'].astype("category")
print(clinical_mod.info())
x1 = clinical_mod.groupby('AJCC Stage', observed=True)


#step 2-calculate average overall survival time for each stage
deaths_sum = x1['OS event'].sum()
tot_patients_count = x1['AJCC Stage'].count()
mean_OS_time = x1['OS Time'].mean()

#Assign these values using `.map()`
clinical_mod = clinical_mod.assign(
    deaths=clinical_mod['AJCC Stage'].map(deaths_sum),
    tot_patients=clinical_mod['AJCC Stage'].map(tot_patients_count),
    meanOSTime=clinical_mod['AJCC Stage'].map(mean_OS_time)
)


clinical_mod = clinical_mod[['AJCC Stage', 'deaths', 'tot_patients', 'meanOSTime']]

clinical_mod_subset = clinical_mod[1:5]
clinical_mod_clean = clinical_mod.dropna()
print(clinical_mod_clean.head(10))  # Data after dropping NaNs

# Group by 'AJCC Stage' and compute mean values
clinical_summary = clinical_mod_clean.groupby('AJCC Stage', observed=True, as_index=False).mean()

# Display the cleaned dataset
print(clinical_summary)

#step 3 - calculate difference between average survival time between AJCC stages 
# Sort by meanOS Time in descending order (highest survival time first)
clinical_summary = clinical_summary.sort_values(by="AJCC Stage")

# Calculate the difference in mean OS Time between consecutive stages
clinical_summary['survival_time_diff'] = clinical_summary['meanOSTime'].diff(periods=-1)  # Show decrease

# Display the cleaned and correctly ordered dataset
print(clinical_summary)

#step 4-printing the greatest differences between stages 
def find_greatest_survival_difference(df):
    max_idx = df['survival_time_diff'].abs().idxmax()
    return df.loc[max_idx, ['AJCC Stage', 'survival_time_diff']].to_dict()

# Run the function
greatest_diff = find_greatest_survival_difference(clinical_summary)

# Print result
print(f"\n Greatest Difference in Survival Time: {greatest_diff['survival_time_diff']:.2f} days in **{greatest_diff['AJCC Stage']}**")

#Analysis: 
#The results indicate significant differences in OS Time across AJCC stages, confirming 
#that disease progression impacts survival time.Stage I patients have the highest survival time (906.67 days), while Stage IIIB has the lowest (316.67 days), 
#showing a sharp decline as cancer advances.The largest survival time drop (435.08 days) occurs between Stage 
#IIIA and Stage IIIB, highlighting a critical transition point.
#Interestingly, Stage IIB has a higher survival time than Stage III (961.7 vs. 562 days), suggesting possible treatment effects or 
#sample size variations. Stage IV (802 days) has better survival than some lower stages, which 
#might indicate advanced treatments extending survival even in late-stage cases.
#Yes, there are significant differences in OS Time across AJCC stages. Survival time generally decreases with disease 
#progression, with the most drastic declines occurring at later stages. Further analysis could explore treatment effects and patient 
#characteristics influencing survival variations.



hotel_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/hotels.csv")
print(hotel_data)

## Question 3:
#Does a longer lead time cause a booking to be cancelled? In other words, does longer lead time increase the liklihood 
#of a booking being cancelled. 
#question 3 type: Causal Inference 
lead_max = hotel_data["lead_time"].max()
print(lead_max)
lead_min = hotel_data["lead_time"].min()
print(lead_min)
#Pseudocode
#1-group by lead times 
#2-group by cancellation liklihood where 0 means not cancelled and 1 means cancelled 
#3-within each lead time, compute the number and percentage of cancellations (0s)
#4-calculate the lead time range with the highest cancellation rate 

#step 1 -group by lead times 
hotel_data["lead_time"] = pd.to_numeric(hotel_data["lead_time"])
# Define bins and labels to categorize lead times into different ranges
bins = [-1, 100, 200, 300, 400, 500, 600, 700, 800]
labels = ["0-100 days", "100-200 days", "200-300 days", "300-400 days", "400-500 days", "500-600 days", "600-700 days", "700-800 days"]
# Categorize bookings based on their lead time
hotel_data["lead_time1"] = pd.cut(hotel_data["lead_time"], bins=bins, labels=labels)
# Sort and display the last 10 entries to verify the binning process
sorted_output = hotel_data[["lead_time", "lead_time1"]].sort_values(by="lead_time", ascending=True)
print(sorted_output.tail(10))
# Count the number of bookings in each lead time category
lead_time_counts = hotel_data["lead_time1"].value_counts()
print(lead_time_counts)
#the result gives the number of bookings made within the range of labeled dates 


#step 2 -group by cancellation liklihood where 0 means not cancelled and 1 means cancelled 
is_canceled_counts = hotel_data.groupby("is_canceled").size()
print(is_canceled_counts)

#step 3 -within each lead time, compute the number and percentage of cancellations (0s)
lead_canceled_table = hotel_data.groupby(["lead_time1", "is_canceled"]).size().unstack()
print(lead_canceled_table)

#calculating the percent of cancellations within each lead time day range 
lead_canceled_percent = lead_canceled_table.div(lead_canceled_table.sum(axis=1), axis=0) * 100

# Display the percentage table
print(lead_canceled_percent)


#step 4 -calculate the lead time range with the highest cancellation liklihood 
def highest_cancellation_lead_time(lead_canceled_percent):
    highest_canceled_lead_time = lead_canceled_percent.idxmax(axis=1)
    return highest_canceled_lead_time
highest_canceled_lead_time_result = highest_cancellation_lead_time(lead_canceled_percent)
print(highest_canceled_lead_time_result)

"""
    Function to determine the lead time range with the highest likelihood of cancellation.

    Parameters:
    lead_canceled_percent (DataFrame): A DataFrame containing the percentage of cancellations 
                                       for each lead time category.

    Returns:
    Series: The lead time range with the highest cancellation likelihood.
    """

#Analysis 
#The data indicates that while cancellation rates vary across different lead 
#times, there is no clear causal relationship between lead time and the 
#likelihood of cancellation. If a direct causal link existed, we would expect 
#a consistent trend—either an increase or decrease in cancellation rates as lead time grows. 
#However, the data shows fluctuations: cancellations are high (72.2%) for 0-100 days, decrease for
# 200-400 days, then rise again to 100% for 600-700 days, only to drop to 0% for 700-800 days. 
#These inconsistencies suggest that other unobserved factors, such as pricing policies, seasonality, customer type (e.g., business vs. 
#leisure travelers), or external events, may be influencing cancellations. The fact that some long lead times have high cancellations while others 
#have none implies that lead time alone does not determine whether a booking will be canceled. Instead, cancellations may be driven by a complex 
#interplay of multiple variables, making it difficult to establish a direct cause-and-effect relationship between lead time and cancellation 
#likelihood.



#Another question I want to explore in this dataset is:
#Question 4: Can a guest's nationality predict their preferred meal plan
#question 4 type: Predictive modelling
#I used this week's class learnings and sample code 

#Pseudocode:
#1-Prepare the data by removing missing values and encoding categorical variables.
#2-Split the dataset into predictors (X) and target variable (y).
#3-Train a Linear Regression model to predict meal plans based on country.
#4-Evaluate the model’s performance using Mean Squared Error (MSE). 

#step 1-Prepare the data by removing missing values and encoding categorical variables.
hotel_data = hotel_data[['meal', 'country']].dropna()  # Drop missing values
hotel_data['meal'] = hotel_data['meal'].astype('category').cat.codes  # Encode meal as numeric
hotel_data = pd.get_dummies(hotel_data, columns=['country'])  # One-hot encode country column

# Step 2: Define predictors (X) and target variable (y)
X = hotel_data.drop(columns=['meal'])  # Select all columns except 'meal' as predictors
y = hotel_data['meal']  # Target variable (meal preference)

#step 3- Train a Linear Regression model to predict meal plans based on country.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


#step 4- Evaluate the model’s performance using Mean Squared Error (MSE). 
from sklearn.metrics import mean_squared_error

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print MSE
print(f"Mean Squared Error: {mse:.2f}")


#analysis:The Mean Squared Error (MSE) of 1.12 indicates that the model's 
#predictions for meal plans based on a guest's country are not highly accurate. 
#Since meal plan choices are categorical, Linear Regression may not be the best model for this problem, 
#as it assumes a continuous relationship between the variables. The relatively high MSE suggests that 
#a guest’s nationality alone is not a strong predictor of their meal plan preference, and other 
#factors—such as trip purpose, group size, or booking channel—might influence meal choices more significantly. 
#A better approach could involve classification models like Logistic Regression or Decision Trees which 
#are more suitable for categorical predictions.

