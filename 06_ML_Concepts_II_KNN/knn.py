"""
K-Nearest Neighbors (KNN) Classification with Nested Cross-Validation

This script demonstrates KNN classification using nested cross-validation:
- Outer loop: Evaluates model performance on test set
- Inner loop: Tunes hyperparameters using validation set

The code follows PEP 8 standards and includes extensive comments for
educational purposes.

Author: DS-3021 Course
Date: 2024
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns

# Scikit-learn imports for machine learning
from sklearn.model_selection import (cross_val_score, GridSearchCV, KFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# Import our custom data preparation functions
from data_preparation import (prepare_bank_data, split_train_val_test,
                              prepare_features_and_target)


# ============================================================================
# LOAD AND EXPLORE DATA
# ============================================================================
def load_and_explore_data():
    """
    Load the bank marketing dataset and perform initial exploration.

    This function loads data from a remote URL and prints summary statistics
    to help understand the dataset structure and composition.

    Returns:
    --------
    pandas.DataFrame
        The raw bank marketing dataset
    """
    print("=" * 80)
    print("LOADING AND EXPLORING DATA")
    print("=" * 80)

    # Load data from GitHub repository
    url = "https://raw.githubusercontent.com/UVADS/DS-3001/main/data/bank.csv"
    bank_data = pd.read_csv(url)

    # Display basic information about the dataset
    print("\nDataset Info:")
    print(bank_data.info())

    # Examine the composition of categorical variables
    print("\n" + "-" * 80)
    print("CATEGORICAL VARIABLE DISTRIBUTIONS")
    print("-" * 80)

    categorical_vars = ['marital', 'education', 'default', 'job', 'contact',
                        'housing', 'poutcome', 'signed up']

    for var in categorical_vars:
        if var in bank_data.columns:
            print(f"\n{var.upper()}:")
            print(bank_data[var].value_counts())

    return bank_data


# ============================================================================
# CHECK FOR MISSING DATA
# ============================================================================
def check_missing_data(df):
    """
    Visualize missing data in the dataset.

    This function creates a visualization showing the proportion of missing
    values for each variable. Understanding missingness is crucial before
    modeling.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to check for missing values
    """
    print("\n" + "=" * 80)
    print("CHECKING FOR MISSING DATA")
    print("=" * 80)

    # Create a visualization of missing data
    sns.displot(
        data=df.isna().melt(value_name="missing"),
        y="variable",
        hue="missing",
        multiple="fill",
        aspect=1.25
    )
    plt.title("Missing Data Visualization")

    # Count missing values per column
    missing_counts = df.isna().sum()
    if missing_counts.sum() == 0:
        print("\n✓ No missing data found in the dataset!")
    else:
        print("\nMissing values per column:")
        print(missing_counts[missing_counts > 0])

    plt.show()


# ============================================================================
# NESTED CROSS-VALIDATION FUNCTION
# ============================================================================
def nested_cross_validation(X, y, k_range=range(1, 22, 2),
                            outer_cv_folds=5, inner_cv_folds=3,
                            random_state=1984):
    """
    Perform nested cross-validation for hyperparameter tuning and evaluation.

    NESTED CROSS-VALIDATION STRUCTURE:
    - Outer Loop: Splits data into training and test sets for unbiased
                  performance evaluation
    - Inner Loop: Within each outer fold, performs cross-validation on the
                  training set to select the best hyperparameter (k)

    This approach prevents overfitting and provides a more realistic estimate
    of model performance on unseen data.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    k_range : range, default=range(1, 22, 2)
        Range of k values to test (number of neighbors)
    outer_cv_folds : int, default=5
        Number of folds for outer cross-validation loop
    inner_cv_folds : int, default=3
        Number of folds for inner cross-validation loop
    random_state : int, default=1984
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'outer_scores': List of scores from outer CV
        - 'best_k_per_fold': Best k value found in each outer fold
        - 'mean_score': Mean score across all outer folds
        - 'std_score': Standard deviation of scores
    """
    print("\n" + "=" * 80)
    print("NESTED CROSS-VALIDATION")
    print("=" * 80)
    print(f"\nOuter CV Folds: {outer_cv_folds}")
    print(f"Inner CV Folds: {inner_cv_folds}")
    print(f"Testing k values: {list(k_range)}")

    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    # Create outer cross-validation splitter
    outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True,
                     random_state=random_state)

    # Lists to store results from each outer fold
    outer_scores = []
    best_k_per_fold = []

    # OUTER LOOP: Iterate through each fold for final evaluation
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        print(f"\n{'-' * 80}")
        print(f"OUTER FOLD {fold_num}/{outer_cv_folds}")
        print(f"{'-' * 80}")

        # Split data into training and test sets for this outer fold
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # INNER LOOP: Use GridSearchCV to find the best k
        # GridSearchCV automatically performs cross-validation to tune
        # hyperparameters
        inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True,
                         random_state=random_state)

        # Create parameter grid for k values
        param_grid = {'n_neighbors': list(k_range)}

        # Initialize KNN classifier
        knn = KNeighborsClassifier()

        # Perform grid search with inner cross-validation
        print("\nPerforming inner CV to find best k...")
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1  # Use all available CPU cores
        )

        # Fit grid search on training data
        grid_search.fit(X_train_outer, y_train_outer)

        # Get the best k value from inner CV
        best_k = grid_search.best_params_['n_neighbors']
        best_k_per_fold.append(best_k)

        print(f"Best k selected by inner CV: {best_k}")
        print(f"Inner CV score for best k: {grid_search.best_score_:.4f}")

        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_
        outer_score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(outer_score)

        print(f"Outer fold test score: {outer_score:.4f}")

    # Calculate final statistics across all outer folds
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)

    print("\n" + "=" * 80)
    print("NESTED CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nScores from each outer fold: {outer_scores}")
    print(f"Best k selected in each fold: {best_k_per_fold}")
    print(f"\nMean outer CV score: {mean_score:.4f} "
          f"(+/- {std_score:.4f})")
    most_common_k = max(set(best_k_per_fold), key=best_k_per_fold.count)
    print(f"Most common best k: {most_common_k}")

    return {
        'outer_scores': outer_scores,
        'best_k_per_fold': best_k_per_fold,
        'mean_score': mean_score,
        'std_score': std_score
    }


# ============================================================================
# TRAIN FINAL MODEL AND EVALUATE
# ============================================================================
def train_and_evaluate_final_model(X_train, y_train, X_val, y_val, X_test,
                                   y_test, k_range=range(1, 22, 2),
                                   cv_folds=5, random_state=1984):
    """
    Train the final model using the best k found via cross-validation.

    This function:
    1. Uses cross-validation on training data to select best k
    2. Trains final model with best k on all training data
    3. Evaluates on validation set
    4. Provides final evaluation on test set

    Parameters:
    -----------
    X_train, y_train : array-like
        Training features and target
    X_val, y_val : array-like
        Validation features and target
    X_test, y_test : array-like
        Test features and target
    k_range : range
        Range of k values to test
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    sklearn model
        The trained KNN model with best k
    int
        The best k value selected
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH CROSS-VALIDATION")
    print("=" * 80)

    # Set random seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    # Use cross-validation to find the best k on training data
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Store cross-validation scores for each k
    cv_scores = []

    print(f"\nTesting k values: {list(k_range)}")
    print(f"Using {cv_folds}-fold cross-validation on training set\n")

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Perform cross-validation and get accuracy scores
        scores = cross_val_score(knn, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        cv_scores.append(mean_score)
        print(f"k={k:2d}: CV Score = {mean_score:.4f} "
              f"(+/- {scores.std():.4f})")

    # Find the k with the highest cross-validation score
    best_k = list(k_range)[np.argmax(cv_scores)]
    best_cv_score = max(cv_scores)

    print(f"\n{'=' * 80}")
    print(f"BEST K SELECTED: {best_k} (CV Score: {best_cv_score:.4f})")
    print(f"{'=' * 80}")

    # Train final model with best k on all training data
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)

    # Evaluate on training set (should be high, but watch for overfitting)
    train_score = final_model.score(X_train, y_train)
    print(f"\nTraining Accuracy: {train_score:.4f}")

    # Evaluate on validation set
    val_score = final_model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score:.4f}")

    # Evaluate on test set (final unbiased evaluation)
    test_score = final_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_score:.4f}")

    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), cv_scores, marker='o', linestyle='-',
             linewidth=2)
    plt.axvline(x=best_k, color='r', linestyle='--',
                label=f'Best k={best_k}')
    plt.xlabel('Number of Neighbors (k)', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title('KNN Model Performance vs. k Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return final_model, best_k


# ============================================================================
# DETAILED MODEL EVALUATION
# ============================================================================
def evaluate_model_detailed(model, X_test, y_test, X_val, y_val):
    """
    Provide detailed evaluation metrics and visualizations.

    This function creates:
    - Confusion matrix visualization
    - Classification report (precision, recall, F1-score)
    - ROC curve and AUC score
    - Various performance metrics

    Parameters:
    -----------
    model : sklearn model
        Trained KNN model
    X_test, y_test : array-like
        Test features and target
    X_val, y_val : array-like
        Validation features and target
    """
    print("\n" + "=" * 80)
    print("DETAILED MODEL EVALUATION")
    print("=" * 80)

    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)

    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # ---- CONFUSION MATRIX ----
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX (Validation Set)")
    print("-" * 80)

    cm = confusion_matrix(y_val, y_val_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Validation Set')
    plt.tight_layout()
    plt.show()

    # ---- CLASSIFICATION REPORT ----
    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("-" * 80)
    print(classification_report(y_val, y_val_pred))

    # Calculate sensitivity and specificity manually
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
    specificity = tn / (tn + fp)  # True Negative Rate

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # ---- ROC CURVE AND AUC ----
    print("\n" + "-" * 80)
    print("ROC CURVE AND AUC SCORE")
    print("-" * 80)

    # Calculate ROC curve and AUC for test set
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:, 1])
    auc = metrics.roc_auc_score(y_test, y_test_proba[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nAUC Score (Test Set): {auc:.4f}")

    # ---- F1 SCORE ----
    f1_val = metrics.f1_score(y_val, y_val_pred)
    f1_test = metrics.f1_score(y_test, y_test_pred)
    print(f"\nF1 Score (Validation): {f1_val:.4f}")
    print(f"F1 Score (Test): {f1_test:.4f}")

    # ---- LOG LOSS ----
    log_loss_val = metrics.log_loss(y_val, y_val_proba)
    log_loss_test = metrics.log_loss(y_test, y_test_proba)
    print(f"\nLog Loss (Validation): {log_loss_val:.4f}")
    print(f"Log Loss (Test): {log_loss_test:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function that orchestrates the entire KNN analysis pipeline.

    This function:
    1. Loads and explores the data
    2. Checks for missing values
    3. Prepares and preprocesses the data
    4. Splits into train/validation/test sets
    5. Performs nested cross-validation
    6. Trains final model with best hyperparameters
    7. Provides detailed evaluation metrics
    """
    print("\n" + "=" * 80)
    print("KNN CLASSIFICATION WITH NESTED CROSS-VALIDATION")
    print("=" * 80)

    # Step 1: Load and explore data
    bank_data = load_and_explore_data()

    # Step 2: Check for missing data
    check_missing_data(bank_data)

    # Step 3: Prepare data using our reusable functions
    print("\n" + "=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)
    print("\nApplying data preparation pipeline:")
    print("1. Collapsing job categories")
    print("2. Converting to categorical types")
    print("3. Normalizing numeric features")
    print("4. One-hot encoding categorical variables")

    prepared_data, scaler = prepare_bank_data(bank_data)
    print("\n✓ Data preparation complete!")
    print(f"  Original shape: {bank_data.shape}")
    print(f"  Prepared shape: {prepared_data.shape}")

    # Step 4: Split into train, validation, and test sets
    print("\n" + "=" * 80)
    print("DATA SPLITTING")
    print("=" * 80)

    train, test, val = split_train_val_test(
        prepared_data,
        'signed up_1',
        test_size=0.4,
        val_size=0.5,
        random_state=1984
    )

    print(f"\nTraining set size: {len(train)} samples")
    print(f"Validation set size: {len(val)} samples")
    print(f"Test set size: {len(test)} samples")

    # Calculate and display class balance
    prevalence = prepared_data['signed up_1'].sum() / len(prepared_data)
    print(f"\nTarget prevalence (baseline): {prevalence:.4f}")
    print("This represents the probability of randomly selecting a positive")
    print("case. Our model should perform better than this baseline.")

    # Step 5: Prepare features and targets
    X_train, y_train = prepare_features_and_target(train, 'signed up_1')
    X_val, y_val = prepare_features_and_target(val, 'signed up_1')
    X_test, y_test = prepare_features_and_target(test, 'signed up_1')

    # Step 6: Perform nested cross-validation
    # This provides an unbiased estimate of model performance
    nested_cv_results = nested_cross_validation(
        X=pd.concat([X_train, X_val]),  # Use train+val for outer CV
        y=np.concatenate([y_train, y_val]),
        k_range=range(1, 22, 2),
        outer_cv_folds=5,
        inner_cv_folds=3,
        random_state=1984
    )

    # Step 7: Train final model using standard approach
    # Use training set for k selection, validate on validation set,
    # and provide final evaluation on test set
    final_model, best_k = train_and_evaluate_final_model(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        k_range=range(1, 22, 2),
        cv_folds=5,
        random_state=1984
    )

    # Step 8: Detailed model evaluation
    evaluate_model_detailed(final_model, X_test, y_test, X_val, y_val)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print(f"1. Best k value: {best_k}")
    print(f"2. Nested CV mean score: {nested_cv_results['mean_score']:.4f}")
    print("3. Model demonstrates proper generalization to unseen data")
    print("4. Both inner and outer CV loops help prevent overfitting")

    return final_model, nested_cv_results


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
if __name__ == "__main__":
    # Execute the main analysis pipeline
    model, results = main()
