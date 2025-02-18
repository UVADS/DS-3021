#lab dataset2.py
import pandas as pd 
import numpy as np 

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

