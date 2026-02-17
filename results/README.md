# Results Directory

This directory contains visualizations and outputs from the analysis.

## Generated Files

When you run the analysis, the following files will be created:

- `alpha_performance.png` - MSE, MAE, and R² vs alpha curves for all models
- `test_performance.png` - Bar charts comparing test set metrics
- `feature_importance.png` - Feature coefficient comparison across models
- `predictions_vs_actual.png` - Scatter plots of predicted vs actual values
- `residuals.png` - Residual plots for model diagnostics

## Generating Results

Run the complete analysis:

```bash
python run_analysis.py
```

Or using the module directly:

```bash
python src/visualization.py
```

All visualizations will be saved to this directory.
