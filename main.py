'''
M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho. (2021)
"Early prediction of student’s performance in higher education: a case study" 
Trends and Applications in Information Systems and Technologies, vol.1, 
in Advances in Intelligent Systems and Computing series.
Springer. DOI: 10.1007/978-3-030-72657-7_16
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def one_hot_encode_categorical(df, categorical_columns, drop_first=False):
    # Create a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()
    
    # Apply one-hot encoding to each categorical column
    for col in categorical_columns:
        # Get dummies (one-hot encoding)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        
        # Add the dummies to the dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Drop the original column
        df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded

def scale_numerical(df, numerical_columns, oversample=False):
    # Create a copy of the dataframe to avoid modifying the original
    df_scaled = df.copy()
    
    scaler = StandardScaler()
    # Fit the scaler on the training data
    scaler.fit(df[numerical_columns])
    
    # Transform the data
    scaled_values = scaler.transform(df[numerical_columns])

    if oversample:
        ros = RandomOverSampler()
        df_scaled = ros.fit_resample(df_scaled, df_scaled['Target'])
    
    # Replace the original columns with scaled values
    df_scaled[numerical_columns] = scaled_values
    
    return df_scaled, scaler

def oversample_data(X, y):
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    return X_resampled, y_resampled

def preprocess_and_split(df, categorical_columns, numerical_columns, oversample=False):
    df_copy = df.copy()

    df_encoded = one_hot_encode_categorical(df_copy, categorical_columns)
    df_processed, scaler = scale_numerical(df_encoded, numerical_columns)
    
    # Split into train, validation, and test sets
    train, valid, test = np.split(df_processed.sample(frac=1), [int(0.8*len(df_processed)), int(0.9*len(df_processed))])

    # Split into features and target
    X_train = train.drop('Target', axis=1)
    y_train = train['Target']
    X_valid = valid.drop('Target', axis=1)
    y_valid = valid['Target']
    X_test = test.drop('Target', axis=1)
    y_test = test['Target']
    
    # Oversample the training set if requested
    if oversample:
        X_train, y_train = oversample_data(X_train, y_train)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler

def train_svm_with_different_kernels(X_train, y_train, X_valid, y_valid):
    """
    Train SVM with different kernels and compare performance.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix
    y_train : pandas.Series or numpy.ndarray
        Training target variable
    X_valid : pandas.DataFrame or numpy.ndarray
        Validation feature matrix
    y_valid : pandas.Series or numpy.ndarray
        Validation target variable
        
    Returns:
    --------
    dict
        Dictionary containing trained SVM models with different kernels
    """
    # Define different kernels to test
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    
    svm_results = {}
    best_model = None
    best_score = 0
    
    print("\n" + "="*50)
    print("TRAINING SVM WITH DIFFERENT KERNELS")
    print("="*50)
    
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")
        
        # Create SVM model
        svm_model = SVC(kernel=kernel, random_state=42, probability=True)
        
        # Train the model
        svm_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = svm_model.predict(X_valid)
        
        # Calculate metrics
        accuracy = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred, average='weighted')
        
        print(f"SVM ({kernel}) - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # Store results
        svm_results[kernel] = {
            'model': svm_model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = kernel
    
    print(f"\nBest SVM kernel: {best_model} with F1 Score: {best_score:.4f}")
    
    return svm_results, best_model

def hyperparameter_tuning_svm(X_train, y_train, X_valid, y_valid, kernel='rbf'):
    """
    Perform hyperparameter tuning for SVM using GridSearchCV.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix
    y_train : pandas.Series or numpy.ndarray
        Training target variable
    X_valid : pandas.DataFrame or numpy.ndarray
        Validation feature matrix
    y_valid : pandas.Series or numpy.ndarray
        Validation target variable
    kernel : str, default='rbf'
        Kernel to use for hyperparameter tuning
        
    Returns:
    --------
    sklearn.svm.SVC
        Best SVM model after hyperparameter tuning
    """
    print(f"\n" + "="*50)
    print(f"HYPERPARAMETER TUNING FOR SVM ({kernel} kernel)")
    print("="*50)
    
    # Define parameter grid based on kernel
    if kernel == 'rbf':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    elif kernel == 'linear':
        param_grid = {
            'C': [0.1, 1, 10, 100]
        }
    elif kernel == 'poly':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    else:  # sigmoid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    
    # Create SVM model
    svm_model = SVC(kernel=kernel, random_state=42, probability=True)
    
    # Perform grid search
    print("Performing grid search...")
    grid_search = GridSearchCV(
        svm_model, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm = grid_search.best_estimator_
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    y_pred = best_svm.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted')
    
    print(f"\nValidation set performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return best_svm

def evaluate_svm_detailed(model, X_test, y_test, model_name="SVM"):
    """
    Perform detailed evaluation of SVM model.
    
    Parameters:
    -----------
    model : sklearn.svm.SVC
        Trained SVM model
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature matrix
    y_test : pandas.Series or numpy.ndarray
        Test target variable
    model_name : str, default="SVM"
        Name of the model for display purposes
    """
    print(f"\n" + "="*50)
    print(f"DETAILED EVALUATION OF {model_name}")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    target_names = ['Dropout', 'Graduate', 'Enrolled']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
    else:
        # Use matplotlib imshow for confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        # Add text annotations
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        
        plt.xticks(range(len(target_names)), target_names)
        plt.yticks(range(len(target_names)), target_names)
    
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Class-wise performance
    print(f"\nClass-wise Performance:")
    for i, class_name in enumerate(target_names):
        class_accuracy = cm[i, i] / cm[i, :].sum()
        print(f"{class_name}: {class_accuracy:.4f}")

def compare_svm_models(svm_results, X_test, y_test):
    """
    Compare different SVM models on test set.
    
    Parameters:
    -----------
    svm_results : dict
        Dictionary containing SVM models with different kernels
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature matrix
    y_test : pandas.Series or numpy.ndarray
        Test target variable
    """
    print(f"\n" + "="*50)
    print("COMPARISON OF SVM MODELS ON TEST SET")
    print("="*50)
    
    test_results = {}
    
    for kernel, result in svm_results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        test_results[kernel] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"SVM ({kernel}) - Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}")
    
    # Find best model on test set
    best_test_model = max(test_results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest model on test set: SVM ({best_test_model[0]}) with F1: {best_test_model[1]['f1_score']:.4f}")
    
    return test_results

if __name__ == '__main__':
    # Read data
    df = pd.read_csv("data.csv", sep=";")

    # Convert df['Target'] into numbers
    target_mapping = {
        'Dropout': 0,
        'Graduate': 1,
        'Enrolled': 2
    }
    df['Target'] = df['Target'].map(target_mapping)

    # Define categorical and numerical columns
    categorical_columns = [
        'Marital status', 'Application mode', 'Application order', 'Course', 
        'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality',
        'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation',
        'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
    ]
    numerical_columns = [
        'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]

    # Process and split the data
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = preprocess_and_split(
        df, categorical_columns, numerical_columns, oversample=True)
    
    # Print shapes of datasets
    print(f"\nFinal dataset shapes:")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_valid.shape[0]} samples, {X_valid.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # 1. Train SVM with different kernels
    svm_results, best_kernel = train_svm_with_different_kernels(X_train, y_train, X_valid, y_valid)
    
    # 2. Perform hyperparameter tuning on the best kernel
    best_svm_tuned = hyperparameter_tuning_svm(X_train, y_train, X_valid, y_valid, kernel=best_kernel)
    
    # 3. Compare all SVM models on test set
    test_results = compare_svm_models(svm_results, X_test, y_test)
    
    # 4. Detailed evaluation of the best tuned model
    evaluate_svm_detailed(best_svm_tuned, X_test, y_test, f"Best SVM ({best_kernel} - Tuned)")
    