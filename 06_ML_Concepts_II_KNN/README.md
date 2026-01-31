# KNN Classification with Nested Cross-Validation

This directory contains updated KNN (K-Nearest Neighbors) classification code that implements nested cross-validation for robust model evaluation and hyperparameter tuning.

## Overview

The code has been restructured into two main files:

1. **`data_preparation.py`** - Reusable data preprocessing functions
2. **`knn.py`** - Main KNN analysis with nested cross-validation

## What's New

### 1. Nested Cross-Validation

The updated code implements a proper nested cross-validation structure:

- **Outer Loop**: Evaluates overall model performance using 5-fold cross-validation
  - Provides unbiased estimate of model performance
  - Test set remains completely independent
  
- **Inner Loop**: Tunes hyperparameters (k values) using 3-fold cross-validation
  - Selects the best k value for each outer fold
  - Prevents information leakage from test set

This approach ensures that hyperparameter selection doesn't bias the final performance estimate.

### 2. Reusable Data Preparation Module

The `data_preparation.py` module provides clean, well-documented functions for:

- **Data cleaning**: Collapsing categorical levels
- **Feature engineering**: One-hot encoding categorical variables
- **Normalization**: Scaling numeric features using MinMax scaling
- **Data splitting**: Creating stratified train/validation/test splits

All functions include:
- Comprehensive docstrings
- Type hints for parameters
- Usage examples
- Educational comments

### 3. PEP 8 Compliance

Both files fully comply with PEP 8 Python style guidelines:
- Maximum line length of 79 characters
- Proper indentation and spacing
- Clear, descriptive variable names
- Consistent formatting throughout

### 4. Enhanced Documentation

The code includes extensive comments explaining:
- Why each step is necessary
- How algorithms work
- What students should learn
- Best practices in machine learning

## Files Description

### `data_preparation.py`

Contains the following functions:

- `collapse_job_categories()` - Simplifies job categories to reduce dimensionality
- `convert_to_categorical()` - Converts columns to categorical data type
- `normalize_numeric_columns()` - Scales numeric features to [0, 1] range
- `one_hot_encode_categorical()` - Converts categorical variables to binary columns
- `prepare_bank_data()` - Complete preprocessing pipeline for bank marketing data
- `split_train_val_test()` - Creates stratified train/validation/test splits
- `prepare_features_and_target()` - Separates features from target variable

### `knn.py`

Main analysis script with the following structure:

1. **Data Loading and Exploration**
   - Loads bank marketing dataset
   - Displays summary statistics
   - Checks for missing values

2. **Data Preparation**
   - Uses functions from `data_preparation.py`
   - Creates clean, normalized dataset

3. **Nested Cross-Validation**
   - Implements proper nested CV for unbiased evaluation
   - Tests multiple k values (1, 3, 5, ..., 21)
   - Reports best k for each fold

4. **Final Model Training**
   - Trains model with best k on training data
   - Validates on validation set
   - Evaluates on test set

5. **Detailed Evaluation**
   - Confusion matrix visualization
   - Classification report (precision, recall, F1)
   - ROC curve and AUC score
   - Sensitivity and specificity

## How to Use

### Basic Usage

Simply run the main script:

```python
python knn.py
```

This will:
1. Load the bank marketing dataset
2. Preprocess the data
3. Perform nested cross-validation
4. Train and evaluate the final model
5. Display comprehensive results and visualizations

### Using Data Preparation Functions Separately

You can import and use the data preparation functions in your own code:

```python
from data_preparation import prepare_bank_data, split_train_val_test
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Prepare the data
prepared_data, scaler = prepare_bank_data(data)

# Split into train/val/test
train, test, val = split_train_val_test(prepared_data, 'target_column')
```

### Customizing the Analysis

You can modify key parameters in the `main()` function:

```python
# Change k values to test
nested_cv_results = nested_cross_validation(
    X=X_train_val,
    y=y_train_val,
    k_range=range(1, 31, 2),  # Test different k values
    outer_cv_folds=10,         # More folds for larger datasets
    inner_cv_folds=5,          # More inner folds for better tuning
    random_state=1984
)
```

## Educational Benefits

This code is designed for students learning machine learning. It demonstrates:

1. **Proper Cross-Validation**: Shows why nested CV is important
2. **Code Organization**: Separates concerns into reusable modules
3. **PEP 8 Standards**: Models professional Python coding practices
4. **Comprehensive Documentation**: Every step is explained
5. **Best Practices**: Follows ML best practices throughout

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install dependencies:
```bash
pip install -r ../requirements.txt
```

## Key Concepts Demonstrated

### Nested Cross-Validation

Traditional single cross-validation can overestimate performance when used for both hyperparameter tuning and model evaluation. Nested CV solves this by:

1. **Outer loop**: Estimates generalization performance
2. **Inner loop**: Selects hyperparameters

This ensures the test set truly represents unseen data.

### Stratified Splitting

All data splits maintain the class distribution of the original dataset, which is crucial for imbalanced datasets like the bank marketing data (only ~11.6% positive cases).

### Feature Scaling

Distance-based algorithms like KNN are sensitive to feature scales. The code normalizes all numeric features to [0, 1] range using MinMax scaling.

### One-Hot Encoding

Categorical variables are converted to binary columns so they can be used in distance calculations. The code handles this automatically.

## Performance Metrics

The code reports multiple metrics:

- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are correct
- **Recall (Sensitivity)**: Of actual positives, how many are detected
- **Specificity**: Of actual negatives, how many are detected
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve, overall discriminative ability

## Original vs. Updated Code

### Original KNN.py

- Single train/test split
- Manual k selection using loop
- Limited comments
- PEP 8 violations
- All code in one file
- No proper cross-validation for evaluation

### Updated Code

- ✅ Nested cross-validation implementation
- ✅ PEP 8 compliant
- ✅ Modular design with reusable functions
- ✅ Extensive educational comments
- ✅ Proper evaluation methodology
- ✅ Professional code organization

## Common Questions

**Q: Why combine train and validation sets for nested CV?**

A: Nested CV creates its own internal splits. By giving it train+val data, we maximize the data available for hyperparameter tuning while keeping the test set completely independent. The outer loop splits train+val into training and validation portions.

**Q: Why use both nested CV and traditional train/val/test?**

A: The code shows both approaches for educational purposes. Nested CV provides the most robust performance estimate, while the traditional split demonstrates a common workflow students will encounter.

**Q: Which k value should I use?**

A: The nested CV will report which k values perform best. Common choices are odd numbers (to avoid ties) between 1 and 21. The code tests these automatically.

## License

This code is part of the DS-3021 course materials.

## Authors

DS-3021 Course Team
University of Virginia - School of Data Science
