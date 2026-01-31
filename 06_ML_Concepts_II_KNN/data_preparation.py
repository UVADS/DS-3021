"""
Data Preparation Module for Machine Learning

This module contains reusable functions for data cleaning, preprocessing,
and preparation that can be used across different machine learning projects.
All functions follow PEP 8 standards and include detailed documentation.
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def collapse_job_categories(df, job_column='job'):
    """
    Collapse job categories into employed/unemployed categories.

    This function simplifies the 'job' column by grouping multiple job types
    into two main categories: 'Employed' and 'Unemployed'. This reduces the
    number of categorical levels and can improve model performance.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the job column
    job_column : str, default='job'
        The name of the column containing job categories

    Returns:
    --------
    pandas.DataFrame
        DataFrame with simplified job categories

    Examples:
    ---------
    >>> df = collapse_job_categories(bank_data)
    >>> print(df['job'].value_counts())
    """
    # Define which job types are considered employed
    employed = ['admin', 'blue-collar', 'entrepreneur', 'housemaid',
                'management', 'self-employed', 'services', 'technician']

    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Apply the categorization using a lambda function
    df_copy[job_column] = df_copy[job_column].apply(
        lambda x: "Employed" if x in employed else "Unemployed"
    )

    return df_copy


def convert_to_categorical(df, columns):
    """
    Convert specified columns to categorical data type.

    Categorical data types are memory efficient and required for many ML
    operations. This function converts multiple columns at once.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list
        List of column names to convert to categorical type

    Returns:
    --------
    pandas.DataFrame
        DataFrame with specified columns converted to categorical

    Examples:
    ---------
    >>> cat_cols = ['job', 'marital', 'education']
    >>> df = convert_to_categorical(bank_data, cat_cols)
    """
    df_copy = df.copy()
    df_copy[columns] = df_copy[columns].astype('category')
    return df_copy


def normalize_numeric_columns(df, method='minmax'):
    """
    Normalize numeric columns using MinMax or Standard scaling.

    Normalization scales numeric features to a similar range, which is
    important for distance-based algorithms like KNN. MinMax scaling
    transforms features to [0, 1] range, while Standard scaling transforms
    to mean=0 and std=1.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    method : str, default='minmax'
        Scaling method: 'minmax' or 'standard'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized numeric columns
    sklearn scaler object
        The fitted scaler for potential inverse transformation

    Examples:
    ---------
    >>> normalized_df, scaler = normalize_numeric_columns(bank_data)
    >>> print(normalized_df.describe())
    """
    df_copy = df.copy()

    # Select only numeric columns (int64 or float64)
    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns

    # Choose the appropriate scaler
    if method == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif method == 'standard':
        scaler = preprocessing.StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")

    # Fit and transform the numeric columns
    scaled_data = scaler.fit_transform(df_copy[numeric_cols])

    # Convert back to DataFrame with original column names
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols,
                             index=df_copy.index)

    # Replace the numeric columns in the original dataframe
    df_copy[numeric_cols] = scaled_df

    return df_copy, scaler


def one_hot_encode_categorical(df, categorical_columns=None):
    """
    One-hot encode categorical variables.

    One-hot encoding converts categorical variables into a format that can
    be provided to ML algorithms. Each category becomes a binary column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    categorical_columns : list, optional
        List of categorical column names. If None, auto-detects categorical
        columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one-hot encoded categorical variables

    Examples:
    ---------
    >>> encoded_df = one_hot_encode_categorical(bank_data)
    >>> print(encoded_df.columns)
    """
    df_copy = df.copy()

    # Auto-detect categorical columns if not specified
    if categorical_columns is None:
        categorical_columns = (df_copy.select_dtypes(include='category')
                               .columns.tolist())

    # Perform one-hot encoding
    encoded = pd.get_dummies(df_copy[categorical_columns])

    # Drop original categorical columns and join encoded ones
    df_copy = df_copy.drop(categorical_columns, axis=1)
    df_copy = df_copy.join(encoded)

    return df_copy


def prepare_bank_data(df, target_column='signed up'):
    """
    Complete data preparation pipeline for bank marketing dataset.

    This function performs all necessary data preparation steps in the
    correct order:
    1. Collapse job categories
    2. Convert to categorical types
    3. Normalize numeric features
    4. One-hot encode categorical features

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw bank marketing dataset
    target_column : str, default='signed up'
        Name of the target variable column

    Returns:
    --------
    pandas.DataFrame
        Fully preprocessed dataset ready for modeling
    sklearn scaler object
        The fitted scaler for potential use on new data

    Examples:
    ---------
    >>> prepared_data, scaler = prepare_bank_data(bank_data)
    >>> print(prepared_data.shape)
    """
    # Step 1: Collapse job categories to reduce dimensionality
    df_processed = collapse_job_categories(df)

    # Step 2: Define and convert categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing',
                'contact', 'poutcome', target_column]
    df_processed = convert_to_categorical(df_processed, cat_cols)

    # Step 3: Normalize numeric columns for distance-based algorithms
    df_processed, scaler = normalize_numeric_columns(df_processed)

    # Step 4: One-hot encode categorical variables
    df_processed = one_hot_encode_categorical(df_processed)

    return df_processed, scaler


def split_train_val_test(df, target_column, test_size=0.4, val_size=0.5,
                         random_state=1984):
    """
    Split data into training, validation, and test sets with stratification.

    This function creates three datasets:
    - Training set: Used to train the model
    - Validation set: Used to tune hyperparameters
    - Test set: Used for final model evaluation

    Stratification ensures that the class distribution is maintained across
    all splits, which is important for imbalanced datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed dataframe
    target_column : str
        Name of the target variable column (must end with '_1' after encoding)
    test_size : float, default=0.4
        Proportion of data to reserve for test+validation (0 to 1)
    val_size : float, default=0.5
        Proportion of test+validation data to use for validation (0 to 1)
    random_state : int, default=1984
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (train, test, val) - Three DataFrames for training, testing, and
        validation

    Examples:
    ---------
    >>> train, test, val = split_train_val_test(prepared_data,
    ...                                          'signed up_1')
    >>> print(f"Train: {len(train)}, Test: {len(test)}, Val: {len(val)}")
    """
    # First split: separate training from (test + validation)
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state
    )

    # Second split: separate test from validation
    test, val = train_test_split(
        test,
        test_size=val_size,
        stratify=test[target_column],
        random_state=random_state
    )

    return train, test, val


def prepare_features_and_target(df, target_column):
    """
    Separate features (X) from target variable (y).

    This function splits the dataframe into feature matrix and target vector,
    which is the standard format required by scikit-learn models.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing both features and target
    target_column : str
        Name of the target variable column

    Returns:
    --------
    tuple
        (X, y) - Feature matrix and target vector

    Examples:
    ---------
    >>> X_train, y_train = prepare_features_and_target(train,
    ...                                                 'signed up_1')
    >>> print(f"Features: {X_train.shape}, Target: {y_train.shape}")
    """
    X = df.drop([target_column], axis=1)
    y = df[target_column].values

    return X, y
