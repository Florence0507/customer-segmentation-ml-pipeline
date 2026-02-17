"""
Data Preprocessing Module
Handles data loading, encoding, and standardization for regularized regression analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='../data/job_satisfaction_data.csv'):
    """
    Load the job satisfaction dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """
    Print basic exploratory statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to explore
    """
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print("="*60)


def encode_categorical(df):
    """
    Apply one-hot encoding to categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with categorical variables
        
    Returns:
    --------
    pd.DataFrame
        Encoded dataset
    """
    df_encoded = pd.get_dummies(
        df, 
        columns=['Gender', 'Education_Level'], 
        drop_first=True
    )
    print(f"✓ Categorical encoding complete: {len(df_encoded.columns)} total features")
    return df_encoded


def split_data(df, target_col='Job_Satisfaction', test_size=0.2, val_size=0.5, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Encoded dataset
    target_col : str
        Name of target variable
    test_size : float
        Proportion for test+validation combined (default: 0.2 = 20%)
    val_size : float
        Proportion of test_size for validation (default: 0.5 = 50% of 20% = 10%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    print(f"\n✓ Data split complete:")
    print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(X_train, X_val, X_test):
    """
    Standardize features using StandardScaler (fit on training data only).
    
    Parameters:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier interpretation
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("\n✓ Feature standardization complete (mean=0, std=1)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath='../data/job_satisfaction_data.csv', verbose=True):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to dataset
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    dict
        Dictionary containing all processed data and objects
    """
    if verbose:
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
    
    # Load data
    df = load_data(filepath)
    
    # Explore (optional)
    if verbose:
        explore_data(df)
    
    # Encode categorical
    df_encoded = encode_categorical(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    
    # Standardize
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        X_train, X_val, X_test
    )
    
    if verbose:
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }


if __name__ == "__main__":
    # Example usage
    data = preprocess_pipeline()
    print(f"Ready for modeling with {len(data['feature_names'])} features")
