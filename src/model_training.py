"""
Model Training & Evaluation Module
Trains and evaluates Lasso, Ridge, and Elastic Net regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_models_with_alphas(X_train, y_train, X_val, y_val, alpha_values):
    """
    Train and evaluate models across multiple alpha values.
    
    Parameters:
    -----------
    X_train, X_val : pd.DataFrame
        Scaled feature matrices
    y_train, y_val : pd.Series
        Target variables
    alpha_values : list
        List of alpha values to test
        
    Returns:
    --------
    pd.DataFrame
        Validation results for all models and alphas
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: Testing alpha values")
    print("="*70)
    
    results = []
    
    for alpha in alpha_values:
        print(f"\nTraining with alpha = {alpha}...")
        
        # Initialize models
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        ridge = Ridge(alpha=alpha, random_state=42)
        elastic_net = ElasticNet(alpha=alpha, random_state=42, max_iter=10000)
        
        # Train models
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        elastic_net.fit(X_train, y_train)
        
        # Predictions on validation set
        y_pred_lasso = lasso.predict(X_val)
        y_pred_ridge = ridge.predict(X_val)
        y_pred_elastic = elastic_net.predict(X_val)
        
        # Calculate metrics for each model
        for model_name, y_pred in [
            ('Lasso', y_pred_lasso),
            ('Ridge', y_pred_ridge),
            ('ElasticNet', y_pred_elastic)
        ]:
            results.append({
                'Alpha': alpha,
                'Model': model_name,
                'MSE': mean_squared_error(y_val, y_pred),
                'MAE': mean_absolute_error(y_val, y_pred),
                'R²': r2_score(y_val, y_pred)
            })
    
    results_df = pd.DataFrame(results)
    print("\n✓ Hyperparameter tuning complete")
    return results_df


def find_best_alphas(validation_results):
    """
    Find best alpha for each model based on lowest MSE.
    
    Parameters:
    -----------
    validation_results : pd.DataFrame
        Results from train_models_with_alphas
        
    Returns:
    --------
    dict
        Best alpha for each model
    """
    best_alphas = {}
    
    print("\n" + "="*70)
    print("BEST ALPHA VALUES (based on validation MSE)")
    print("="*70)
    
    for model_name in ['Lasso', 'Ridge', 'ElasticNet']:
        model_data = validation_results[validation_results['Model'] == model_name]
        best_row = model_data.loc[model_data['MSE'].idxmin()]
        best_alphas[model_name] = best_row['Alpha']
        
        print(f"\n{model_name}:")
        print(f"  Best Alpha: {best_row['Alpha']}")
        print(f"  MSE: {best_row['MSE']:.4f}")
        print(f"  MAE: {best_row['MAE']:.4f}")
        print(f"  R²: {best_row['R²']:.4f}")
    
    return best_alphas


def train_final_models(X_train, y_train, best_alphas):
    """
    Train final models with best alpha values.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    best_alphas : dict
        Best alpha for each model
        
    Returns:
    --------
    dict
        Trained models
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODELS")
    print("="*70)
    
    models = {
        'Lasso': Lasso(alpha=best_alphas['Lasso'], random_state=42, max_iter=10000),
        'Ridge': Ridge(alpha=best_alphas['Ridge'], random_state=42),
        'ElasticNet': ElasticNet(alpha=best_alphas['ElasticNet'], random_state=42, max_iter=10000)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"✓ {name} trained with alpha = {best_alphas[name]}")
    
    return models


def evaluate_on_test(models, X_test, y_test, best_alphas):
    """
    Evaluate final models on test set.
    
    Parameters:
    -----------
    models : dict
        Trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    best_alphas : dict
        Best alpha for each model
        
    Returns:
    --------
    pd.DataFrame
        Test set performance metrics
    """
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Alpha': best_alphas[name],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        
        print(f"\n{name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    return pd.DataFrame(results)


def get_feature_importance(models, feature_names):
    """
    Extract and compare feature coefficients.
    
    Parameters:
    -----------
    models : dict
        Trained models
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pd.DataFrame
        Feature coefficients for each model
    """
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Lasso': models['Lasso'].coef_,
        'Ridge': models['Ridge'].coef_,
        'ElasticNet': models['ElasticNet'].coef_
    })
    
    # Sort by absolute Ridge coefficient
    coefficients['abs_ridge'] = coefficients['Ridge'].abs()
    coefficients = coefficients.sort_values('abs_ridge', ascending=False)
    coefficients = coefficients.drop('abs_ridge', axis=1)
    
    return coefficients


def full_training_pipeline(data, alpha_values=[0.01, 0.1, 1, 10, 100]):
    """
    Complete training and evaluation pipeline.
    
    Parameters:
    -----------
    data : dict
        Preprocessed data from preprocessing.preprocess_pipeline()
    alpha_values : list
        Alpha values to test
        
    Returns:
    --------
    dict
        All results including models, metrics, and coefficients
    """
    # Hyperparameter tuning
    validation_results = train_models_with_alphas(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        alpha_values
    )
    
    # Find best alphas
    best_alphas = find_best_alphas(validation_results)
    
    # Train final models
    models = train_final_models(
        data['X_train'], data['y_train'],
        best_alphas
    )
    
    # Test evaluation
    test_results = evaluate_on_test(
        models, data['X_test'], data['y_test'],
        best_alphas
    )
    
    # Feature importance
    coefficients = get_feature_importance(models, data['feature_names'])
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    
    return {
        'models': models,
        'validation_results': validation_results,
        'test_results': test_results,
        'coefficients': coefficients,
        'best_alphas': best_alphas
    }


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    
    print("Loading and preprocessing data...")
    data = preprocess_pipeline()
    
    print("\nTraining models...")
    results = full_training_pipeline(data)
    
    print("\n\nFINAL TEST RESULTS:")
    print(results['test_results'].to_string(index=False))
    
    print("\n\nFEATURE COEFFICIENTS:")
    print(results['coefficients'].to_string(index=False))
