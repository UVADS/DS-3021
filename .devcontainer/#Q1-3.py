'''
linear regression lab 
Q0 - answers 
1. A model is "linear" if the output (dependent variable) is a linear combination of the unknown parameters 
(coefficients). This means the equation can be written as Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε, where each 
term involves a coefficient multiplied by a predictor, but no coefficients are squared, multiplied together, 
or inside functions like exponents or logarithms. The term "linear" refers to the linearity of parameters (β’s) appear 
in the equation, not necessarily the variables (X’s).

2- With an intercept, the dummy variable coefficient represents a difference from the baseline. This is because the category represented by 
X=0 exists and its effect is absorbed into the intercept (β0) making the dummy variable coefficient (β1)
measure the difference relative to that baseline. 
Without an intercept, the coefficients represent the absolute values for each category separately.This is because 
each category gets its own coefficient, and there is no baseline group to compare against—instead of measuring 
differences, the model directly assigns predicted values to each category.

3- Linear regression is not suited for classification because it predicts continuous values rather than 
discrete class labels, making it unreliable for distinguishing between categories. It does not naturally 
create decision boundaries or provide probability estimates, leading to overlapping and ambiguous predictions. 

4- High training accuracy but low testing accuracy indicates overfitting, where the model memorizes specific 
patterns within the training data instead of learning general patterns. This happens when the model is too 
complex, capturing noise rather than meaningful relationships. A visible pattern in the residuals suggests the 
model is not properly generalizing and may be missing key underlying trends.

5- There are two ways to handle non-linear relationships in linear regression. The first is feature engineering, 
where we create new features by adding polynomial terms (e.g.x^2, x^3) or interaction terms (e.g., x1 * x2). 
This allows a linear model to fit curves while still remaining linear in how it estimates coefficients. For 
example, instead of using y = β0 + β1x  we can use y=β0 + β1x + β2x^2 to better capture patterns in the data. 
The second method is kernel-based techniques, such as Support Vector Regression (SVR) or Kernel Ridge Regression. 
These models map the original features into a higher-dimensional space, where a simple linear model can find 
complex patterns. Unlike feature engineering, kernel methods do this transformation automatically, without 
manually creating polynomial terms. Both approaches help linear models handle non-linear relationships, but 
one does it explicitly (feature engineering) while the other does it implicitly (kernels).

6- The intercept represents the predicted value of the dependent variable when all independent variables are 
zero. A slope coefficient shows how much the dependent variable changes for a one-unit increase in the 
corresponding independent variable, assuming all other variables remain constant. The coefficient for a 
dummy variable represents the difference in the dependent variable relative to the baseline category 
(when the dummy variable is 0).
'''

#Q0-3
#Q1:
### part 1:
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

q1_clean = pd.read_csv("https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv")
print(q1_clean.head())

#remove extra spaces from column names
q1_clean.columns = q1_clean.columns.str.strip()
# Group by 'Neighbourhood' and calculate the mean price
avg_price = q1_clean.groupby("Neighbourhood")["Price"].mean()

#Print the result
#print(avg_price)
#Group by 'Neighbourhood' and calculate the mean scores
avg_score = q1_clean.groupby("Neighbourhood")["Review Scores Rating"].mean()
print(avg_score)

#most expensive borough
print(avg_price.nlargest(1))

q1_clean["Log_Price"] = np.log(q1_clean["Price"] + 1)

# Set plot size
plt.figure(figsize=(14, 6))

# Kernel Density Plot for Price
plt.subplot(1, 2, 1)
sns.kdeplot(data=q1_clean, x="Price", hue="Neighbourhood", common_norm=False)
plt.title("Kernel Density Plot of Price by Neighborhood")
plt.xlim(0, q1_clean["Price"].max() + 50)

# Kernel Density Plot for Log(Price)
plt.subplot(1, 2, 2)
sns.kdeplot(data=q1_clean, x="Log_Price", hue="Neighbourhood", common_norm=False)
plt.title("Kernel Density Plot of Log(Price) by Neighborhood")

# Display the plots

#plt.show()

### part 2:
# One-hot encode 'Neighbourhood' without dropping any category (since we want no intercept)
# One-hot encode 'Neighbourhood' (no drop_first because we want NO baseline category)
q1_encoded = pd.get_dummies(q1_clean, columns=["Neighbourhood"], drop_first=False)

# Select all dummy variables for neighborhoods (independent variables)
X = q1_encoded.filter(like="Neighbourhood")

# Target variable (dependent variable)
y = q1_clean["Price"]

# Fit linear regression WITHOUT intercept
model = sm.OLS(y, X).fit()

# Print regression results
print(model.summary())

#printing average price from part 1 
print(avg_price)

#identifying the pattern:
#The regression output shows the coefficients for each Neighborhood when predicting Price, without an intercept.
#Since I removed the intercept, each coefficient represents the average price for that specific neighborhood.
#My observation is that regression coefficients are exactly the same as the mean prices from Part 1.
#This happens because, without an intercept, each category (Neighborhood) gets its own independent prediction.
#If I had included an intercept, one neighborhood would act as a baseline, and other coefficients would 
#represent differences from that baseline.

#"What are the coefficients in a regression of a continuous variable on one categorical variable?"
#If you include an intercept, the coefficients represent differences from a baseline category.
#If you remove the intercept, the coefficients represent the actual mean of the dependent variable (Price) 
#for each category (Neighborhood).

### part 3:
#I will handle dummy variables differently by dropping one category- this dropped category becomes the baseline
#In this case, I will drop the first neighbourhood which is bronx

# Drop 'Neighbourhood_Bronx' to set it as the baseline
if "Neighbourhood_Bronx" in q1_encoded.columns:
    X = q1_encoded.drop(columns=["Neighbourhood_Bronx"])
else:
    raise ValueError("ERROR: 'Neighbourhood_Bronx' column is missing! Check one-hot encoding.")

#Ensure X contains only numeric columns (drop any non-numeric columns like 'Property Type' & 'Room Type')
X = X.select_dtypes(include=['number'])

# Convert target variable (Price) to numeric
y = pd.to_numeric(q1_clean["Price"], errors='coerce')

# Drop any rows with missing values
X.dropna(inplace=True)
y = y.loc[X.index]

# Add an intercept
X = sm.add_constant(X)

# Convert to NumPy arrays (Fixes StatsModels dtype issue)
X = X.to_numpy()
y = y.to_numpy()

# Fit regression model
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())

#The intercept represents the average price for the baseline category (in this case, Bronx). Each coefficient 
#for the remaining dummy variables shows how much higher or lower the average price is for that neighborhood 
#compared to the Bronx.

# In a regression without an intercept, each coefficient equals the category’s mean price. To get those values
# from the current model, I simply add the intercept (baseline mean) to each coefficient; that sum gives me the
#absolute mean price for that neighborhood.

### part 4:
sample_size = len(q1_clean)
print(sample_size)
# Define independent variables (X) and dependent variable (y)
q1_encoded = pd.get_dummies(q1_clean, columns=["Neighbourhood"], drop_first=True)  # Drop one category to avoid multicollinearity
X = q1_encoded[["Review Scores Rating"] + list(q1_encoded.filter(like="Neighbourhood").columns)]
y = q1_clean["Price"]

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm the split
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Check data types in X_train
print("Column Data Types in X_train:\n", X_train.dtypes)

# Convert boolean (True/False) to numeric (0/1)
X_train = X_train.astype(int)
X_test = X_test.astype(int)

# Add an intercept to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Now Fit the Regression Model
model = sm.OLS(y_train, X_train).fit()

# Print the summary
print(model.summary())

#R^2 of training set is 0.051 as indicated in the regression output
#I will calculate R^2 of the test set
# Predict prices on the test set
y_pred_test = model.predict(X_test)

# Compute R² on the test set
r2_test = model.rsquared  # This gives R² for training, so we need to compute it for testing

from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred_test)

#print(f"R² on test set: {r2_test:.4f}")

#RMSE on test set
# Compute RMSE on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"RMSE on test set: {rmse_test:.4f}")


#extracting the coeffecient of the review scores rating from the model:
review_score_coef = model.params["Review Scores Rating"]
print(f"Coefficient on Review Scores Rating: {review_score_coef:.4f}")
#In the regression output, the review scores rating is 1.2119 
#This means that for every 1-unit increase in Review Scores Rating, the Price increases by $1.21.

#finding the most expensive property to rent
most_expensive_property = q1_clean.groupby("Property Type")["Price"].mean().idxmax()
print(f"The most expensive kind of property you can rent is: {most_expensive_property}")
#answer: The most expensive kind of property you can rent is: Condominium

### Part 5:
q1_encoded = pd.get_dummies(q1_clean, columns=["Neighbourhood", "Property Type"], drop_first=True)

# Define independent variables (X) and dependent variable (y)
X = q1_encoded[["Review Scores Rating"] + list(q1_encoded.filter(like="Neighbourhood").columns) + list(q1_encoded.filter(like="Property Type").columns)]
y = q1_clean["Price"]
# Split the data into 80% training and 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Convert boolean (True/False) to numeric (0/1)
X_train = X_train.astype(int)
X_test = X_test.astype(int)

# Add an intercept
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the regression model
model = sm.OLS(y_train, X_train).fit()

# Print the summary
print(model.summary())

# Predict prices on the test set
y_pred_test = model.predict(X_test)

# Compute R² on the test set
r2_test = r2_score(y_test, y_pred_test)

# Compute RMSE on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"R² on test set: {r2_test:.4f}")
print(f"RMSE on test set: {rmse_test:.4f}")

#coeffecient for review scores rating
#from the linear regression table of values it is 1.2010

#The most expensive property type is:
most_expensive_property = q1_clean.groupby("Property Type")["Price"].mean().idxmax()
print(f"The most expensive kind of property you can rent is: {most_expensive_property}")

### Part 6:
#coeffecient of review scores rating in part 4 is  1.2119 
#coeffecient of review scores rating from part 5 is 1.2010
#In Part 4:  Review Scores Rating had a coefficient of 1.2119, meaning each 1-point increase in review rating 
#increased price by $1.21, without controlling for Property Type.
#In Part 5 → The coefficient dropped to 1.2010, meaning each 1-point increase in review rating now increases 
#price by $1.20, but now we control for Property Type.
#The slight drop means that some of the effect of Review Scores Rating on price was actually due to different 
#property types having different ratings.
#insight that connects to multiple linear regression: As we add more variables to a multiple linear regression 
#model, the coefficient for a variable can change because the model now accounts for more factors that 
#influence the dependent variable (Price).

#===================================================================================================================

#Q2:
csv_url = 'https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cars_hw.csv'

# Read the CSV file
cars_data = pd.read_csv(csv_url)

# Display the first 5 rows
print(cars_data.head())

### Part 1: 
# Check data types and missing values
print("\nData Types:\n", cars_data.dtypes)
print("\nMissing Values:\n", cars_data.isnull().sum())

# Summary statistics to check for outliers and skewed data
print("\nSummary Statistics:\n", cars_data.describe())

# Visualizing outliers and skewness
plt.figure(figsize=(12, 5))
sns.boxplot(data=cars_data.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Variables (Checking for Outliers)")
plt.show()

#The dataset contains potential outliers in Price, with a maximum of 2,941,000, while the 75th percentile is 883,000, 
#suggesting extremely high values that may distort analysis. Mileage_Run also has potential outliers, with a
# maximum of 99,495 km and a minimum of 1,117 km, which may require investigation and transformation. 

#Both Price and Mileage_Run show right-skewed distributions, as their means are significantly higher than their 
#medians, so applying a log transformation can help normalize them. Seating Capacity is mostly uniform at 5 
#seats, but since some vehicles have up to 8 seats, we should check whether this affects pricing. 883,000, 
#suggesting extremely high values that may distort analysis. Mileage_Run also has potential outliers, with a
# maximum of 99,495 km and a minimum of 1,117 km, which may require investigation and transformation. Both 
#Price and Mileage_Run show right-skewed distributions, as their means are significantly higher than their 
#medians, so applying a log transformation can help normalize them. Seating Capacity is mostly uniform at 5 
# seats, but since some vehicles have up to 8 seats, we should check whether this affects pricing.


#Thus, I will remove extreme outliers based on IQR (Interquartile Range).
# Define numeric columns to check for outliers
num_cols = ["Price", "Mileage_Run"]

# Compute IQR for each column
Q1 = cars_data[num_cols].quantile(0.25)
Q3 = cars_data[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove extreme outliers
cars_cleaned = cars_data[~((cars_data[num_cols] < lower_bound) | (cars_data[num_cols] > upper_bound)).any(axis=1)]

print(f"Rows before cleaning: {len(cars_data)}, Rows after cleaning: {len(cars_cleaned)}")

#Price and Mileage_Run are strictly positive and right-skewed: I will use Log Transformation (log(x+1)).
# Apply log transformation to reduce skewness
cars_cleaned["Price_Log"] = np.log1p(cars_cleaned["Price"])  # log(1+x) to avoid log(0)
cars_cleaned["Mileage_Run_Log"] = np.log1p(cars_cleaned["Mileage_Run"])

# Verify the new distribution after transformation
cars_cleaned[["Price_Log", "Mileage_Run_Log"]].hist(bins=20, figsize=(10, 5))
plt.suptitle("Distributions of Price and Mileage_Run After Log Transformation")
#plt.show()

print(cars_cleaned.head())
#in the cleaned data set, price_log and mileage_run_log were added to help reduce skewness and improve model 
#performance.

### Part 2:
# Convert Price to numeric (force conversion and handle errors)
cars_cleaned["Price"] = pd.to_numeric(cars_cleaned["Price"], errors="coerce")

# Drop any NaN values from Price
cars_cleaned = cars_cleaned.dropna(subset=["Price"])
price_summary = cars_cleaned["Price"].describe()
print(price_summary)

plt.figure(figsize=(8, 5))
sns.kdeplot(cars_cleaned["Price"], fill=True, color="blue")
plt.title("Kernel Density Plot of Price")
plt.xlabel("Price")
plt.ylabel("Density")
plt.show()

# Strip extra spaces from column names
cars_cleaned.columns = cars_cleaned.columns.str.strip()
# Group by 'Make' and summarize prices
price_by_make = cars_cleaned.groupby("Make")["Price"].describe()
print(price_by_make)

plt.figure(figsize=(10, 8))
sns.kdeplot(data=cars_cleaned, x="Price", hue="Make", fill=True, common_norm=False)
plt.title("Kernel Density Plot of Price by Make")
plt.xlabel("Price")
plt.ylabel("Density")
plt.legend(title="Make", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

most_expensive_brands = cars_cleaned.groupby("Make")["Price"].mean().sort_values(ascending=False)
#print(most_expensive_brands)
#The most expensive car brands in this dataset are Kia, Jeep, and Mahindra, with average prices of ₹1.37 
#million (₹13.69 lakhs), ₹1.29 million (₹12.9 lakhs), and ₹1.1 million (₹11 lakhs), respectively. These 
#brands are known for producing larger, premium SUVs and off-road vehicles, which contribute to their higher 
#price points. On the other end of the spectrum, Datsun and Chevrolet are the most affordable brands, with 
#average prices of ₹2.89 lakhs and ₹4.53 lakhs, respectively, indicating their focus on budget-friendly models.
#Car prices in general show a wide range, with mid-range brands like Maruti Suzuki (₹5.88 lakhs), Volkswagen 
#(₹6.05 lakhs), and Renault (₹6.62 lakhs) falling between budget and premium categories. The overall price 
#distribution is likely right-skewed, meaning that while most cars are priced lower, a few high-end models 
#significantly increase the average price. This trend is common in the automobile market, where luxury SUVs 
#and high-performance vehicles pull the average price upwards.


### Part 3:
# Split the dataset into 80% training and 20% testing
train_data, test_data = train_test_split(cars_cleaned, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
print(f"Training set size: {train_data.shape[0]} rows")
print(f"Testing set size: {test_data.shape[0]} rows")

### Part 4:
#model 1: regress price on numerical variables only 
print(cars_cleaned.dtypes)
# Select numeric variables again after verification
numeric_features = ["Make_Year", "Mileage_Run_Log", "Seating_Capacity", "Price_Log"]

# Define independent variables (X) and dependent variable (y)
X = cars_cleaned[numeric_features]
y = cars_cleaned["Price"]

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add an intercept to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the OLS regression model using training data
model = sm.OLS(y_train, X_train).fit()

# Predict on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute R² and RMSE for training set
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Compute R² and RMSE for test set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print results
print(f"Training R²: {r2_train:.4f}, Training RMSE: {rmse_train:.2f}")
print(f"Test R²: {r2_test:.4f}, Test RMSE: {rmse_test:.2f}")

#model 2: regress price on categorical variables only
# Define categorical features
categorical_features = ["Make", "Color", "Body_Type", "No_of_Owners", "Fuel_Type", "Transmission", "Transmission_Type"]

# Strip spaces from column names
cars_cleaned.columns = cars_cleaned.columns.str.strip()

# Recheck for missing categorical columns
missing_cols = [col for col in categorical_features if col not in cars_cleaned.columns]
if missing_cols:
    print("Missing categorical columns:", missing_cols)
else:
    print("All categorical columns are available!")

# Filter only the existing categorical columns
existing_categorical_features = [col for col in categorical_features if col in cars_cleaned.columns]

# One-hot encode categorical variables (drop_first=True to avoid dummy variable trap)
if existing_categorical_features:
    X_categorical = pd.get_dummies(cars_cleaned[existing_categorical_features], drop_first=True)
    X_categorical = X_categorical.astype(int)  # Convert to integer
    #print("One-hot encoding successful!")
else:
    #print(" No categorical features found to encode.")
# Define dependent variable (y)
    y = cars_cleaned["Price"]

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_categorical, y, test_size=0.2, random_state=42)

# Add an intercept to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the OLS regression model using training data
model_categorical = sm.OLS(y_train, X_train).fit()

# Predict on test set
y_test_pred = model_categorical.predict(X_test)

# Compute R² and RMSE for test set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print results
#print(f"Test R²: {r2_test:.4f}, Test RMSE: {rmse_test:.2f}")


#Model that performs better on the test set is: 
#Model 1, which uses numerical variables only, outperforms Model 2, which relies solely on categorical 
#variables, based on both Test R² and Test RMSE. Model 1 has a Test R² of 0.9495, meaning it explains 
#94.95% of the variance in price, while Model 2 only explains 52.82% (R² = 0.5282). A higher R² indicates 
#that Model 1 has significantly better predictive power. Additionally, Model 1 has a Test RMSE of ₹60,046.34, 
#meaning the average prediction error is around ₹60,000, compared to ₹183,540.01 for Model 2, which has a 
#much higher error rate. The stronger performance of Model 1 is likely because numerical variables, such as 
#Make_Year and Mileage_Run_Log, carry more predictive information about car prices than categorical features 
#alone. While one-hot encoding categorical variables helps, it does not fully capture the pricing trends. 
#Since Model 2 lacks numerical predictors, it struggles to provide precise estimates. Therefore, Model 1 is 
#the better choice for predicting car prices. 

#model 3:
# Define numerical variables again
numeric_features = ["Make_Year", "Mileage_Run_Log", "Seating_Capacity", "Price_Log"]

# Ensure numerical features exist
X_numerical = cars_cleaned[numeric_features]
categorical_features = ["Make", "Color", "Body_Type", "No_of_Owners", "Fuel_Type", "Transmission", "Transmission_Type"]

# One-hot encode categorical variables (drop_first=True to avoid dummy variable trap)
X_categorical = pd.get_dummies(cars_cleaned[categorical_features], drop_first=True).astype(int)
X_combined = pd.concat([X_categorical, X_numerical], axis=1)
# Define dependent variable (y)
y = cars_cleaned["Price"]

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Add an intercept to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the OLS regression model using training data
model_combined = sm.OLS(y_train, X_train).fit()

# Predict on test set
y_test_pred_combined = model_combined.predict(X_test)

# Compute R² and RMSE for the combined model on the test set
r2_test_combined = r2_score(y_test, y_test_pred_combined)
rmse_test_combined = np.sqrt(mean_squared_error(y_test, y_test_pred_combined))

# Compare performance with previous models
performance_comparison = {
    "Model": ["Numerical Only", "Categorical Only", "Combined Model"],
    "Test R²": [0.9495, 0.5282, r2_test_combined],
    "Test RMSE": [60046.34, 183540.01, rmse_test_combined]
}

# Display performance comparison
performance_df = pd.DataFrame(performance_comparison)
#print(performance_df)
#The combined model, which incorporates both numerical and categorical variables, performs better than both 
#the numerical-only and categorical-only models. It achieves a Test R² of 0.9561, which is 0.66% higher than 
#the numerical-only model (0.9495) and 80.98% higher than the categorical-only model (0.5282), indicating a 
#stronger ability to explain price variations. Additionally, it has the lowest Test RMSE at ₹56,005.89, 
#meaning it makes more accurate predictions compared to the numerical-only model (₹60,046.34, ₹4,040.45 worse) 
#and the categorical-only model (₹183,540.01, ₹127,534.12 worse). While most of the predictive power still 
#comes from numerical features, including categorical variables slightly improves accuracy, making the 
#combined model the best predictor of car prices.

###  Part 5:
max_degree = 20

# Store results for different polynomial degrees
poly_results = []

# Loop over different polynomial degrees
for degree in range(1, max_degree + 1):
    # Apply polynomial feature expansion to numerical variables
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(cars_cleaned[numeric_features])
    
    # Convert expanded features to a DataFrame with proper column names
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=cars_cleaned.index)
    
    # Combine polynomial numerical features with categorical features
    X_combined_poly = pd.concat([X_poly_df, X_categorical], axis=1)
    
    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y, test_size=0.2, random_state=42)
    
    # Add an intercept
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    # Fit the regression model
    model_poly = sm.OLS(y_train, X_train).fit()
    
    # Predict on test set
    y_test_pred_poly = model_poly.predict(X_test)
    
    # Compute R² and RMSE for the test set
    r2_test_poly = r2_score(y_test, y_test_pred_poly)
    rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))
    
    # Store results
    poly_results.append({"Degree": degree, "Test R²": r2_test_poly, "Test RMSE": rmse_test_poly})
    
    # Stop when R² goes negative
    if r2_test_poly < 0:
        break

# Convert results into a DataFrame for display
poly_results_df = pd.DataFrame(poly_results)
print(poly_results_df)

#Your results show that Test R² remains very high (≈1.000) and RMSE keeps decreasing as the polynomial 
#degree increases. However, we do not see Test R² becoming negative, which means the model hasn’t yet 
#started failing in terms of predictive accuracy.

#the best model degree is degree 5 (R² = 1.000, RMSE = 157.50)
#comparison with best model from part 4: combined model (R² = 0.956074, RMSE = 56005.886855)
#The degree 5 polynomial model (R² = 1.000, RMSE = ₹157.50) significantly outperforms the best combined model 
#from Part 4 (R² = 0.9561, RMSE = ₹56,005.89), indicating a nearly perfect fit with minimal prediction error. 
#However, the drastic improvement suggests severe overfitting, meaning the polynomial model is memorizing the 
#training data rather than generalizing well to new data.

### Part 6:
# Predict on test set using the best model (degree 5 polynomial model)
y_test_pred_best = model_poly.predict(X_test)

# Scatter Plot: Predicted vs. True Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linestyle='--')  # Diagonal line
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs. True Prices (Degree 5 Polynomial Model)")
plt.show()

# Compute Residuals
residuals = y_test - y_test_pred_best

# Kernel Density Plot of Residuals
plt.figure(figsize=(8, 6))
sns.kdeplot(residuals, fill=True, color='blue', alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--')  # Vertical line at zero
plt.xlabel("Residuals (Errors)")
plt.ylabel("Density")
plt.title("Kernel Density Plot of Residuals (Degree 5 Polynomial Model)")
plt.show()

# Evaluate Model
model_evaluation = """
### Strengths:
Extremely high R² (1.000) and very low RMSE (₹157.50), indicating a strong fit.
Predictions should align closely with actual values along the diagonal in the scatter plot.

### Weaknesses:
The residuals may not be normally distributed, suggesting overfitting.
Such a perfect R² suggests the model is memorizing training data rather than generalizing well.
High-degree polynomial models are complex and harder to interpret, making them impractical in real-world scenarios.
"""

#print(model_evaluation)

#=========================================================================================================#


#Q3 : 
### Part 1:
#This is a data set on the top instagram influences 
insta_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/top_insta_influencers_data.csv")
print(insta_data)
### Part 2:
# Check for missing values
print(insta_data.isnull().sum())

#since country has 62 missing values, I will put "unknown" in place of missing values
insta_data["country"].fillna("Unknown", inplace=True)
print(insta_data.isnull().sum())

#Display data types of all columns
print(insta_data.dtypes)

# Convert columns if necessary (e.g., removing % signs and converting to numeric)
insta_data["60_day_eng_rate"] = insta_data["60_day_eng_rate"].str.rstrip("%").astype(float) / 100
insta_data["new_post_avg_like"] = insta_data["new_post_avg_like"].str.replace("k", "e3").str.replace("m", "e6").astype(float)
insta_data["total_likes"] = insta_data["total_likes"].str.replace("k", "e3").str.replace("m", "e6").str.replace("b", "e9").astype(float)

print(insta_data.head())  

#get summary statistics
print(insta_data.describe())

# Histogram of Influence Scores
plt.figure(figsize=(8, 5))
sns.histplot(insta_data["influence_score"], bins=20, kde=True)
plt.title("Distribution of Influence Scores")
plt.xlabel("Influence Score")
plt.ylabel("Frequency")
plt.show()


#(rank is a relative measure so I will drop it)
#(channel_info is a unique identifier so is not a meaningful predictor, so I will drop it )

#target/outcome variable: total_likes 
#feature/ predictor variables: 
#numerical= influence_score, new_post_avg_like, 60_day_eng_rate
#categorical= country 

### Part 3:
# Define features (X) and target variable (y)
X = insta_data[["influence_score", "new_post_avg_like", "60_day_eng_rate", "country"]]  # Features
y = insta_data["total_likes"]  # Target variable



#Convert numeric columns to float
numeric_features = ["influence_score", "new_post_avg_like", "60_day_eng_rate"]
insta_data[numeric_features] = insta_data[numeric_features].apply(pd.to_numeric, errors="coerce")

#Handle missing values in numeric columns
insta_data.fillna(0, inplace=True)

#One-hot encode 'country' and convert boolean to int
X_categorical = pd.get_dummies(insta_data["country"], drop_first=True).astype(int)

# Select features (numerical + categorical)
X_numerical = insta_data[["influence_score", "new_post_avg_like", "60_day_eng_rate"]]

#Merge categorical & numerical features
X = pd.concat([X_numerical, X_categorical], axis=1)
y = insta_data["total_likes"].astype(float)  # Ensure y is numeric

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert everything to float (ensures regression works)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

### Part 4:

# Function to run regression and compute RMSE
def run_regression(X_train, X_test, y_train, y_test, model_name):
    X_train = sm.add_constant(X_train)  # Add intercept
    X_test = sm.add_constant(X_test)    # Add intercept

    model = sm.OLS(y_train, X_train).fit()  # Fit the model
    y_test_pred = model.predict(X_test)  # Predict on test set

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Print model summary and RMSE
    print(f"\n {model_name} - Regression Results")
    print(model.summary())
    print(f"Test RMSE: {rmse:.2f}")

# model 1: total_likes ~ influence_score + new_post_avg_like
X_train_1 = X_train[["influence_score", "new_post_avg_like"]]
X_test_1 = X_test[["influence_score", "new_post_avg_like"]]
run_regression(X_train_1, X_test_1, y_train, y_test, "Regression 1")

# model 2: total_likes ~ influence_score + new_post_avg_like + country
X_train_2 = X_train[["influence_score", "new_post_avg_like"] + list(X_train.filter(like="country_").columns)]
X_test_2 = X_test[["influence_score", "new_post_avg_like"] + list(X_test.filter(like="country_").columns)]
run_regression(X_train_2, X_test_2, y_train, y_test, "Regression 2")

# model 3: total_likes ~ influence_score + new_post_avg_like + country + 60_day_eng_rate
X_train_3 = X_train[["influence_score", "new_post_avg_like", "60_day_eng_rate"] + list(X_train.filter(like="country_").columns)]
X_test_3 = X_test[["influence_score", "new_post_avg_like", "60_day_eng_rate"] + list(X_test.filter(like="country_").columns)]
run_regression(X_train_3, X_test_3, y_train, y_test, "Regression 3")

#Model 3 performed the best because it had the lowest Test RMSE (2,897,594,189.35), indicating better prediction accuracy compared to Models 1 and 2. The inclusion of 60_day_eng_rate likely provided additional explanatory power, helping reduce prediction errors. However, the high RMSE values across all models suggest that total_likes may be influenced by other missing factors, or the model might benefit from nonlinear transformations or interaction terms for better performance.

### Part 6:
#I learned that adding more relevant features to a regression model can improve its accuracy, as seen with Model 3 performing the best. However, all models still had very high RMSE values, which means predicting total_likes is difficult with just the available data. This suggests that other important factors, such as engagement type, content quality, or follower demographics, might be missing from the model. 

#Overall, this lab provided a hands-on understanding of linear regression, its strengths, and its limitations in real-world data analysis. We explored how regression models can be used to predict outcomes like car prices or Instagram engagement, highlighting the importance of feature selection, transformations, and model evaluation. One key takeaway was that simply adding more features does not always guarantee better predictions—relevant, well-engineered features matter more than just increasing the number of predictors.


