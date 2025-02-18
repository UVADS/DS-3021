#machine learning bootcamp dataset2 
import pandas as pd
import numpy as np 
#make sure to install sklearn in your terminal first!
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

campus_recruitment = pd.read_csv("/workspaces/DS-3021-analytics-1/data/Placement_Data_Full_Class.csv")
print(campus_recruitment.head())

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


#Question I want to explore with the campus recruitment dataset
#Which degree specialization (Sci&Tech, Comm&Mgmt) is most in demand by corporations, as indicated by placement rates and salary outcomes?


