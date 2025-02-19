#machine learning bootcamp lab
# Imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
#make sure to install sklearn in your terminal first!
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#I am first going to analyse the college completion data set
'''
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


Train = train_pipeline(college_dt)
Test = test_pipeline(college_dt)

# Display dataset sizes
print("Training Set:", Train)
print("Testing Set:", Test)
'''
#--------------------------------------------------------------------------
#------------------------------------------------------------------------
#data set 2:

jobs_data = pd.read_csv("/workspaces/DS-3021-analytics-1/data/Placement_Data_Full_Class.csv")
#print(jobs_data.head())

#Next, I am going to look at the campus recruitment dataset (dataset 2)
#q1- problems that could be addressed with the dataset (dataset 2)
#Does higher education performance (degree_p) have a significant impact on placement and salary?
#What is the relationship between MBA specialization and salary outcomes?
#How does having work experience (workex) influence placement status and salary?
#Do students with prior work experience receive higher salaries than those without?
#Do students specializing in Marketing & Finance (Mkt&Fin) receive higher salaries compared to those in Marketing & HR (Mkt&HR)?
#What are the strongest predictors of high placement salaries in this dataset?
#What is the average salary difference between students with and without work experience?
#Which degree specialization (Sci&Tech, Comm&Mgmt) is most in demand by corporations, as indicated by placement rates and salary outcomes?

#final exploration for phase I
#working to develop a model that predicts salary (salary)
#target variable- salary

#Translation into a Business Context:
#This model helps HR and recruitment teams predict expected salary offers based on a candidate’s 
#academic background, work experience, and skills. It enables companies to make data-driven hiring decisions,
#optimize compensation structures, and identify high-potential candidates.

#Prediction Problem (Classification or Regression):
#Regression: Since salary is a continuous numerical variable, this is a regression problem. The model will 
#predict the exact salary based on input features rather than classifying candidates into predefined salary 
#brackets.

#step 2
#Independent Business Metric:
#High salaries attract top students with exemplary accomplishments and experiences. Hiring top talent enhances
# efficiency, drives better business outcomes, unlocks opportunities, and improves overall performance. 
#Predicting which student factors lead to the highest salaries helps businesses make strategic hiring 
#decisions and maximize talent potential.

#print(jobs_data.info())
#Three Variables to Convert to Categorical for Salary Analysis:
#workex (Work Experience: Yes/No)
#specialisation (MBA Specialization: Mkt&HR vs. Mkt&Fin)
#degree_t (Type of Degree: Sci&Tech vs. Comm&Mgmt)
#grade percentage achieved in mba (mba_p)

#converting these variables into categorical variables:
jobs_data[['degree_t', 'workex', 'specialisation', 'mba_p']] = (
    jobs_data[['degree_t', 'workex', 'specialisation', 'mba_p']].astype(str)
)
jobs_data[['degree_t', 'workex', 'specialisation', 'mba_p']] = (
    jobs_data[['degree_t', 'workex', 'specialisation', 'mba_p']].astype('category')
)

#print(jobs_data.info())

#taking a closer look at specialisation 
num_categories = jobs_data["specialisation"].nunique()
#print(f"Number of categories in specialisation: {num_categories}")

#taking a closer look at work experience 
num_categories = jobs_data["workex"].nunique()
#print(f"Number of categories in workex: {num_categories}")

#grouping mba_percentage scores into categories 
# Define 3 bins based on data distribution
bins = [50, 60, 65, 78]  # Adjusted for three groups
labels = ['50-60', '60-65', '65-78']  # Meaningful categories

# Apply binning to mba_p
jobs_data["mba_p_category"] = pd.cut(
    jobs_data["mba_p"].astype(float), bins=bins, labels=labels, include_lowest=True
)

# Convert to category type
jobs_data["mba_p_category"] = jobs_data["mba_p_category"].astype("category")

# Verify the new column distribution
#print(jobs_data["mba_p_category"].value_counts())

#scaling / centering the data 
# Initialize StandardScaler
scaler = StandardScaler()

# Apply Standardization (Z-score normalization) to `mba_p`
jobs_data['mba_p_scaled'] = scaler.fit_transform(jobs_data[['mba_p']])

# Display the first 10 standardized values
#print(jobs_data['mba_p_scaled'][:10])
# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max Scaling to `mba_p`
jobs_data['mba_p_normalized'] = scaler.fit_transform(jobs_data[['mba_p']])

# Display the first 10 scaled values
#print(jobs_data['mba_p_normalized'][:10])

#plotting to check
# Convert mba_p to float (if needed)
jobs_data['mba_p'] = jobs_data['mba_p'].astype(float)

# Plot density
jobs_data['mba_p'].plot.density()

# Add title and labels
plt.title('Density Plot of MBA Percentage (mba_p)')
plt.xlabel('MBA Percentage')
plt.ylabel('Density')

# Show the plot
#plt.show()
#pd.DataFrame(jobs_data[['mba_p_normalized']]).plot.density()

#Now we can move forward in normalizing the numeric values and create a index based on numeric columns:
# Select numeric columns
numeric_cols = list(jobs_data.select_dtypes('number'))

# Apply Min-Max Scaling
jobs_data[numeric_cols] = MinMaxScaler().fit_transform(jobs_data[numeric_cols])

# Display the dataset with scaled values
#print(jobs_data.head())

#one hot encoding
# Select categorical columns (excluding non-categorical string columns like 'gender')
category_list = list(jobs_data.select_dtypes('category'))

# Step 1: Add 'Unknown' as a valid category for each categorical column
for col in category_list:
    jobs_data[col] = jobs_data[col].cat.add_categories(['Unknown'])

# Step 2: Fill NaN values in categorical columns with 'Unknown'
jobs_data[category_list] = jobs_data[category_list].fillna("Unknown")

# Step 3: Apply one-hot encoding ONLY to valid categorical columns
jobs_data_1h = pd.get_dummies(jobs_data, columns=category_list)

# Step 4: Ensure all remaining NaN values are replaced with 0 (to prevent conversion issues)
jobs_data_1h = jobs_data_1h.fillna(0)

# Step 5: Convert only one-hot encoded **categorical variables** to integers (excluding text columns)
for col in jobs_data_1h.columns:
    if jobs_data_1h[col].dtype == 'bool':  # Convert True/False to 1/0
        jobs_data_1h[col] = jobs_data_1h[col].astype(int)

# Display the transformed dataset
#print(jobs_data_1h.head())
jobs_data_1h.boxplot(column='salary', vert=False, grid=False)
plt.title('Boxplot of Salary')
#plt.show()

# Summary statistics for salary
#print(jobs_data_1h['salary'].describe())  #  upper quartile (Q3)= 0.111486

#adding salary_f as a binary predictor without removing salary
# Define bins based on salary distribution
bins = [-1, 0.111486, 1]  # Lower bound, Q3 (75th percentile), Max
labels = [0, 1]  # 0 = Below Q3, 1 = Top Quartile

# Apply binning to create the binary predictor
jobs_data_1h['salary_f'] = pd.cut(jobs_data_1h['salary'], bins=bins, labels=labels)

# Display the dataset with the new binary column
#print(jobs_data_1h[['salary', 'salary_f']].head())

##So now let's check the prevalence 
# Compute prevalence (percentage of salary_f values that are 1)
prevalence = jobs_data_1h.salary_f.value_counts()[1] / len(jobs_data_1h.salary_f)
#print(prevalence)

#print(jobs_data_1h.salary_f.value_counts())
#print(54/(54+161)) #looks good 

##Divide up our data into three parts, Training, Tuning, and Test but first we need to...
#clean up our dataset a bit by dropping the original rating variable and the cereal name since we can't really use them
jobs_data_dt = jobs_data_1h.drop(['salary'], axis=1)

# Display the cleaned dataset
#print(jobs_data_dt.head())

## Now we partition
Train, Test = train_test_split(jobs_data_dt, train_size=150, stratify=jobs_data_dt.salary_f)
#I chose train size as 150 because there are 215 rows and the train set must take up ~ 70% of the data entries
#print(Train.shape)
#print(Test.shape)

#Now we need to use the function again to create the tuning set
Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test.salary_f)
#check the prevalance in all groups, they should be all equivalent in order to eventually build an accurate model
#print(Train.salary_f.value_counts())
#print(38/(38+112))
#print(Tune.salary_f.value_counts())
#print(8/(8+24))
print(Test.salary_f.value_counts())
print(8/(8+25))
#The slight variation in proportions across Train (0.2533), Tune (0.25), and Test (0.2424) is 
#due to stratified sampling rounding effects, where class distributions are preserved as closely as 
#possible but may not be exactly equal due to the dataset's finite size


#analysis:
#In Phase I of my salary prediction model, I focused on data preprocessing, exploratory analysis, and 
#dataset partitioning to prepare for building an accurate regression model. My goal was to help HR and 
#recruitment teams make data-driven salary predictions based on candidates’ academic background, work 
#experience, and MBA specialization. To achieve this, I converted categorical variables into numerical f
#ormat using one-hot encoding, grouped MBA percentages into meaningful bins, and standardized/normalized 
#numerical features to ensure consistency across the dataset. I also examined the distribution of salary, 
#identifying the top 25% salary threshold (Q3 = 0.111486) and using it to create a binary classification
# (salary_f = 1 for high salaries, salary_f = 0 for others). This provided a baseline prevalence of ~25% 
#for high salaries, meaning that any predictive model I develop must outperform this benchmark to be valuable.

#After cleaning and transforming the data, I split it into Training (150 rows), Tuning (32 rows), and 
#Test (33 rows) sets, making sure the class balance was maintained. While there were small variations in 
#proportions due to rounding, the overall distribution remained stable. The Training set will be used to 
#build the model, the Tuning set will help optimize it, and the Test set will evaluate its final performance. 
#Now that the dataset is fully prepared and I have established a baseline salary prediction 
#(~25% high salaries), I am ready to move into Phase II: building and testing the model.

#step 3
#The dataset offers a foundation for predicting salary, but its small size (215 rows) limits its ability to 
#generalize well to new data. A small dataset increases the risk of overfitting, where the model learns 
#patterns too specific to the training data and struggles with real-world predictions. Additionally, the 
#high salary class (salary_f = 1) makes up only ~25% of the data, creating an imbalance that could affect 
#model accuracy, especially if it struggles to correctly predict higher salaries.

#Another concern that I have is feature importance—while degree type, work experience, and MBA specialization 
#seem relevant, some variables may have little impact on salary, leading to noise in the model. If certain 
#features do not strongly correlate with salary, they could introduce unnecessary complexity without 
#improving predictions. Careful feature selection and engineering will be crucial in the next phase to 
#ensure the model captures meaningful relationships while avoiding irrelevant data.


#step 4
def train_pipeline(df, train_size=150, target_variable='salary_f'):
    """
    Produces the Training dataset with a specified size, ensuring class balance through stratified sampling.
    """
    Train, _ = train_test_split(df, train_size=train_size, stratify=df[target_variable], random_state=42)
    return Train

def test_pipeline(df, test_size=65, target_variable='salary_f'):
    """
    Produces the Test dataset with a specified size, ensuring class balance through stratified sampling.
    """
    _, Test = train_test_split(df, train_size=(len(df) - test_size), stratify=df[target_variable], random_state=42)
    return Test


Train = train_pipeline(jobs_data_dt)
Test = test_pipeline(jobs_data_dt)

# Display dataset sizes
print("Training Set:", Train)
print("Test Set", Test)



















