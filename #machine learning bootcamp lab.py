#machine learning bootcamp lab
# Imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
#make sure to install sklearn in your terminal first!
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#I am first going to analyse the college completion data set

college_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/cc_institution_details.csv")
#print(college_data.head())

#q1- problems that could be addressed with the dataset (dataset 1)
#How do endowment size and financial aid availability correlate with graduation rates across institutions?
#Do schools with larger endowments tend to have higher retention and graduation rates?
#Do public universities outperform private institutions in terms of student retention and graduation rates?
#How do HBCUs compare to non-HBCUs in terms of graduation rates, retention, and financial aid availability?
#Does a higher full-time faculty percentage (ft_fac_value) correlate with better retention and graduation rates?
#Are there states where public institutions consistently outperform private ones in graduation rates?
#Are research universities (classified under basic) outperforming smaller liberal arts or master's colleges?
#How strongly do median SAT scores (med_sat_value) correlate with college graduation rates (grad_100_value, grad_150_value), 
#and does this relationship vary by state or institution type (public vs. private, research vs. liberal arts colleges)?


#understanding 4 key outcome variables:
#grad_100_value = % of students graduating in 4 years.
#grad_100_percentile = How this 4-year rate compares to other schools.
#grad_150_value = % of students graduating in 6 years.
#grad_150_percentile = How this 6-year rate compares to other schools.

#final exploration for phase I
#working to develop a model that can predict students' graduate rates (grad_100_value) 
#target variable-  grad_100_value

# Translation into a business context: 
#predictive model that helps universities, policymakers, investors or potential employers make better decisions.
#Universities could use this model to identify factors that improve graduation rates.
#Policy Makers could use it to compare education quality across states.
#Investors (e.g., in EdTech) could assess which institutions have stronger academic performance.
#employers could hire candidates from top schools to improve worker productivity and improve outcomes for their companies

# prediction problem (classification or regression): 
#Regression because the target variable (grad_100_value) is a continuous numerical value.
#We are predicting a percentage value of graduation rate based on input parameters

#Q2
#Independent business metric: 
#Assuming that higher graduation rates at a university attract more applicants and increase student spending on 
#education, can we predict which student characteristics are most strongly associated with a higher likelihood of 
#graduating within four years?"

#college_data.info()

#convert these 3 variables into categories I want to explore further 
#control- type of instituition = control
#median SAT percentile = med_sat_percentile
#cohort size = cohort_size 


Column_index_list = [6,16,25,62]
college_data.iloc[:,Column_index_list]= college_data.iloc[:,Column_index_list].astype('category') 
#print(college_data.info())

#print(college_data["control"].value_counts()) #looks good 

bins = [0, 20, 40, 60, 80, 100]  # Bin edges
labels = ['10-20', '20-40', '40-60', '60-80', '80-100']  # Corresponding labels

# Bin the numerical values into categories
college_data["med_sat_percentile"] = pd.cut(
    college_data["med_sat_percentile"], bins=bins, labels=labels, include_lowest=True
)

# Convert to category type
college_data["med_sat_percentile"] = college_data["med_sat_percentile"].astype("category")

# Verify the distribution
#print(college_data["med_sat_percentile"].value_counts()) #Looks good now 


bins = [0, 500, 2000, 10000, float("inf")]  # Upper limit for each category
labels = ["Small (1-500)", "Medium (501-2000)", "Large (2001-10000)", "Very Large (10,001+)"]

# Apply binning
college_data["cohort_size_group"] = pd.cut(
    college_data["cohort_size"], bins=bins, labels=labels, include_lowest=True
)

# Convert to categorical
college_data["cohort_size_group"] = college_data["cohort_size_group"].astype("category")

# Verify the distribution
#print(college_data["cohort_size_group"].value_counts()) #looks good now 


#Scaling/ centering the data
# Select numerical columns (excluding categorical ones)
numeric_cols = college_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical columns
college_data_scaled = college_data.copy()  # Keep a copy of the original data
college_data_scaled[numeric_cols] = scaler.fit_transform(college_data[numeric_cols])

# Display the first few rows of scaled data
#print(college_data_scaled.head())
college_data_scaled["cohort_size"][:10]  # First 10 standardized values
college_data_scaled["med_sat_percentile"][:10]

#normalising the numeric columns in my dataset:
#Select numeric columns for normalization
numeric_cols = college_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numeric columns
college_data_normalized = college_data.copy()  # Keep a copy of the original data
college_data_normalized[numeric_cols] = scaler.fit_transform(college_data[numeric_cols])

# Display the first few rows to verify the transformation
print(college_data_normalized.head())
college_data_normalized["cohort_size"][:10] 

#making a graphical plot to ensure relationships are the same 
college_data["cohort_size"] = pd.to_numeric(college_data["cohort_size"], errors="coerce")
college_data_normalized["cohort_size"] = pd.to_numeric(college_data_normalized["cohort_size"], errors="coerce")
# Check if cohort_size is numeric
#print(college_data["cohort_size"].dtype)  # Should output float64 or int64
#print(college_data_normalized["cohort_size"].dtype)  # Should output float64

# Plot Density Distribution of Cohort Size Before & After Normalization
plt.figure(figsize=(8, 5))

college_data["cohort_size"].plot.density(label="Original", linestyle="--")
college_data_normalized["cohort_size"].plot.density(label="Normalized", linestyle="-")

plt.title("Density Plot of Cohort Size (Original vs. Normalized)")
plt.xlabel("Cohort Size")
plt.legend()
#plt.show() #the plot indicates that it implies that the normalization preserved the shape of the distribution, 
#only rescaling the values between 0 and 1 without changing their relative relationships.

#now, I will find all numeric columns in the dataset, apply min-max scaling to scale the values between 0 and 1 and 
#replace original values with the scaled values
# Find all numeric columns in the dataset
numeric_cols = list(college_data.select_dtypes('number'))

# Apply Min-Max Scaling to these columns
college_data[numeric_cols] = MinMaxScaler().fit_transform(college_data[numeric_cols])

# Display the first few rows to check the changes
#print(college_data.head())
#this ensures no feature dominates the other 

#the next step is one-hot encoding which converts categorical 
#variables into binary outcomes
# Find all categorical columns
category_list = list(college_data.select_dtypes('category'))

# Print categorical columns (to verify)
#print("Categorical Columns:", category_list)

college_data_1h = pd.get_dummies(college_data, columns=category_list)

# Display the first few rows to check the transformation
#print(college_data_1h.head())

# a before after summary for new features added 
#print("Before Encoding:", len(category_list), "categorical columns")
#print("After Encoding:", college_data_1h.shape[1], "total columns")


#next step is Baseline/ Prevalence:
#We first plot a boxplot to see the distribution of the target variable 
#which is grad_100_value (4-year graduation rate):
college_data_1h.boxplot(column='grad_100_value', vert=False, grid=False)

# Show the plot
plt.title("Boxplot of 4-Year Graduation Rates")
plt.xlabel("Graduation Rate (grad_100_value)")
#plt.show()

#getting summary statistics 
#print(college_data_1h["grad_100_value"].describe())
#notice the upper quartile of values will be above 0.43650 (75th percentile)

#Instead of replacing grad_100_value, we will create a new binary 
#column (high_grad_rate_f) based on whether the graduation rate is in the top quartile.
# Calculate the 75th percentile threshold for graduation rate
quartile_75 = college_data_1h["grad_100_value"].quantile(0.75)
#print("75th Percentile Threshold:", quartile_75)
#This finds the top 25% cutoff for grad_100_value (similar to the 0.43 threshold in the cereal dataset).
# Create a binary categorical column based on the top quartile threshold
college_data_1h["high_grad_rate_f"] = pd.cut(
    college_data_1h["grad_100_value"], bins=[-1, quartile_75, 100], labels=[0, 1]
)

# Verify the new column
#print(college_data_1h[["grad_100_value", "high_grad_rate_f"]].head(20))

#0 → Graduation rate ≤ 75th percentile
#1 → Graduation rate > 75th percentile (Top 25%)

#calculating prevalence- percentage of universities with a high graduation rate
prevalence = college_data_1h["high_grad_rate_f"].value_counts()[1] / college_data_1h["high_grad_rate_f"].dropna().shape[0]
#print("Corrected Prevalence of High Graduation Rate (Top 25%):", prevalence)

#verifying this manually:
# Count occurrences of 0 and 1 in high_grad_rate_f
#print(college_data_1h["high_grad_rate_f"].value_counts())
#print(867/(867+2600))

#now we will drop variables and partition 
# Create a cleaned-up dataset without unnecessary columns
college_dt = college_data_1h.drop(["grad_100_value"], axis=1)  

# Display the first few rows to verify changes
#print(college_dt.head())
#print(college_dt["high_grad_rate_f"].isna().sum()) 
college_dt = college_dt.dropna(subset=["high_grad_rate_f"])
college_dt["high_grad_rate_f"] = college_dt["high_grad_rate_f"].fillna(0)
# Define the target variable for stratification
target_variable = "high_grad_rate_f"  # Our classification target

# Split the data: Train (55 examples), Test (rest)
from sklearn.model_selection import train_test_split
# Now we partition
Train, Test = train_test_split(college_dt, train_size=55, stratify=college_dt["high_grad_rate_f"])
#print(Train.shape)
#print(Test.shape)

Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test["high_grad_rate_f"], random_state=42)

print(Train["high_grad_rate_f"].value_counts())
print(14/(41+14))
print(Tune["high_grad_rate_f"].value_counts())
print(427/(427+1279)) #its not the same as above - the small difference 
#in output occurs because  stratified splitting preserves class 
#proportions as closely as possible, but rounding errors and integer row 
#distribution cause slight variations, especially with small sample sizes.
print(Test["high_grad_rate_f"].value_counts())
print(426/(426+1280))
#The small difference occurs because stratified splitting preserves class proportions as closely as possible, 
#but rounding errors and integer row distribution cause slight variations, especially with small sample sizes

#The Train (0.2545), Tune (0.2503), and Test (0.2497) values are nearly the same, meaning stratification worked, but small differences happened because the split must assign whole rows

#analysis:
#I carefully preprocessed this dataset to develop a predictive model for university graduation rates (grad_100_value), 
#addressing the research question: Can I predict which student characteristics are most strongly associated 
#with a higher likelihood of graduating within four years? After transforming categorical variables, 
#normalizing numerical data, and one-hot encoding, I converted the graduation rate into a binary classification problem (
#(high_grad_rate_f), where universities in the top 25% graduation rates were labeled as 1 (high) and others 
#as 0 (low). The prevalence of high graduation rates was consistent across Training (0.2545), Tuning (0.2503), 
#and Test (0.2497) sets, confirming that stratification preserved class proportions, although minor rounding 
#differences occurred due to integer row distribution.

#These findings suggest that graduation rates are not randomly distributed but rather follow predictable 
#patterns based on institutional characteristics such as cohort size, SAT percentiles, and control type 
#(public/private). The modeling phase can now leverage this structured dataset to identify key predictors 
#of student success, which can inform universities, policymakers, and investors about institutional 
#performance and student outcomes. The next steps would involve training classification models (e.g., 
#logistic regression, decision trees) to quantify the impact of these predictors and validate performance 
#using the Tuning and Test sets

#step 3 
#My instincts tell me that the data provides valuable insights into factors affecting university 
#graduation rates, but there are potential challenges. While key variables like cohort size, SAT 
#percentiles, and control type are included, I’m concerned about missing values, potential biases 
#in institutional reporting, and whether all relevant predictors are captured. The data is structured 
#well for modeling, but further feature engineering and validation will be crucial to ensure reliable 
#predictions. 

#step 4:
def train_pipeline(df, train_size=55, target_variable="high_grad_rate_f"):
    """
    Produces the Training dataset with a specified size, ensuring class balance through stratified sampling.
    """
    Train, _ = train_test_split(df, train_size=train_size, stratify=df[target_variable], random_state=42)
    return Train

def test_pipeline(df, test_size=65, target_variable="high_grad_rate_f"):
    """
    Produces the Test dataset with a specified size, ensuring class balance through stratified sampling.
    """
    _, Test = train_test_split(df, train_size=(len(df) - test_size), stratify=df[target_variable], random_state=42)
    return Test

# ✅ Example Usage:
Train = train_pipeline(college_dt)
Test = test_pipeline(college_dt)

# Display dataset sizes
print("Training Set Shape:", Train)
print("Testing Set Shape:", Test)




