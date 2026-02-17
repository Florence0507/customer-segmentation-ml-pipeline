"""
Visualization Module
Functions for creating analysis plots and figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_alpha_performance(validation_results, save_path=None):
    """
    Plot MSE, MAE, and R² vs alpha for all models.
    
    Parameters:
    -----------
    validation_results : pd.DataFrame
        Validation results from model training
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['MSE', 'MAE', 'R²']
    colors = {'Lasso': '#FF6B6B', 'Ridge': '#4ECDC4', 'ElasticNet': '#95E1D3'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model in ['Lasso', 'Ridge', 'ElasticNet']:
            model_data = validation_results[validation_results['Model'] == model]
            ax.plot(model_data['Alpha'], model_data[metric], 
                   marker='o', label=model, linewidth=2, 
                   markersize=8, color=colors[model])
        
        ax.set_xscale('log')
        ax.set_xlabel('Alpha (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} vs Alpha (Validation Set)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if metric == 'R²':
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle('Hyperparameter Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_test_performance(test_results, save_path=None):
    """
    Bar charts comparing test set performance.
    
    Parameters:
    -----------
    test_results : pd.DataFrame
        Test results from model evaluation
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(test_results['Model'], test_results[metric], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} on Test Set', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison (Test Set)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_feature_importance(coefficients, save_path=None):
    """
    Horizontal bar chart comparing feature coefficients.
    
    Parameters:
    -----------
    coefficients : pd.DataFrame
        Feature coefficients from models
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(coefficients))
    width = 0.25
    
    ax.barh(x_pos - width, coefficients['Lasso'], width, 
           label='Lasso', alpha=0.8, color='#FF6B6B')
    ax.barh(x_pos, coefficients['Ridge'], width, 
           label='Ridge', alpha=0.8, color='#4ECDC4')
    ax.barh(x_pos + width, coefficients['ElasticNet'], width, 
           label='ElasticNet', alpha=0.8, color='#95E1D3')
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(coefficients['Feature'])
    ax.set_xlabel('Coefficient Value (Standardized)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(models, X_test, y_test, best_alphas, save_path=None):
    """
    Scatter plots of predicted vs actual values.
    
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
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'Lasso': '#FF6B6B', 'Ridge': '#4ECDC4', 'ElasticNet': '#95E1D3'}
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=100, 
                  edgecolors='black', linewidth=1, color=colors[name])
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Job Satisfaction', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Job Satisfaction', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (α = {best_alphas[name]})', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Predicted vs Actual Job Satisfaction (Test Set)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_residuals(models, X_test, y_test, save_path=None):
    """
    Residual plots for model diagnostics.
    
    Parameters:
    -----------
    models : dict
        Trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'Lasso': '#FF6B6B', 'Ridge': '#4ECDC4', 'ElasticNet': '#95E1D3'}
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Scatter plot
        ax.scatter(y_pred, residuals, alpha=0.6, s=100,
                  edgecolors='black', linewidth=1, color=colors[name])
        
        # Zero line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted Job Satisfaction', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Residual Plot', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis (Test Set)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def create_all_visualizations(results, data, output_dir='../results'):
    """
    Generate all analysis visualizations.
    
    Parameters:
    -----------
    results : dict
        Results from training pipeline
    data : dict
        Preprocessed data
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # 1. Alpha performance
    plot_alpha_performance(
        results['validation_results'],
        save_path=f"{output_dir}/alpha_performance.png"
    )
    
    # 2. Test performance
    plot_test_performance(
        results['test_results'],
        save_path=f"{output_dir}/test_performance.png"
    )
    
    # 3. Feature importance
    plot_feature_importance(
        results['coefficients'],
        save_path=f"{output_dir}/feature_importance.png"
    )
    
    # 4. Predictions vs actual
    plot_predictions_vs_actual(
        results['models'],
        data['X_test'],
        data['y_test'],
        results['best_alphas'],
        save_path=f"{output_dir}/predictions_vs_actual.png"
    )
    
    # 5. Residuals
    plot_residuals(
        results['models'],
        data['X_test'],
        data['y_test'],
        save_path=f"{output_dir}/residuals.png"
    )
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    from model_training import full_training_pipeline
    
    print("Loading data and training models...")
    data = preprocess_pipeline(verbose=False)
    results = full_training_pipeline(data)
    
    print("\nCreating visualizations...")
    create_all_visualizations(results, data)
