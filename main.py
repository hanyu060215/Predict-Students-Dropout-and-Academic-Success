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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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
    """
    Oversample the minority classes to balance the dataset.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    tuple
        (Oversampled feature matrix, Oversampled target variable)
    """
    # Initialize the random oversampler
    ros = RandomOverSampler(random_state=42)
    
    # Fit and apply the oversampling
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    return X_resampled, y_resampled

def preprocess_and_split(df, categorical_columns, numerical_columns, oversample=False):
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Encode categorical columns
    df_encoded = one_hot_encode_categorical(df_copy, categorical_columns)
    
    # Scale numerical columns
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

    # Plot histograms for numeric columns
    plot_histograms = False
    if plot_histograms:
        for label in df_processed.columns[21:37]:
            if label == 'Target':
                continue

            plt.hist(df_processed[df_processed['Target'] == 0][label], color='blue', label='Dropout', alpha=0.7, density=True)
            plt.hist(df_processed[df_processed['Target'] == 1][label], color='red', label='Graduate', alpha=0.7, density=True)
            plt.hist(df_processed[df_processed['Target'] == 2][label], color='yellow', label='Enrolled', alpha=0.7, density=True)
            plt.title(label)
            plt.ylabel('Probability')
            plt.xlabel(label)
            plt.legend()
            plt.show()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = preprocess_and_split(
        df, categorical_columns, numerical_columns, oversample=True)
    