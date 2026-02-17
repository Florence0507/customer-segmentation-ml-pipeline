# Regularized Regression Analysis: Job Satisfaction Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning analysis comparing **Lasso**, **Ridge**, and **Elastic Net** regression techniques for predicting employee job satisfaction based on demographic and work-related factors.

## 📊 Project Overview

This project systematically evaluates three regularized regression approaches to predict job satisfaction scores, demonstrating:
- **Model Comparison**: Performance analysis across different regularization techniques
- **Hyperparameter Tuning**: Systematic alpha parameter optimization
- **Feature Importance**: Identification of key drivers of job satisfaction
- **Best Practices**: Professional ML workflow including proper train/validation/test splits

### Key Findings

✅ **Best Model**: Lasso Regression (α = 0.1)  
- **R² Score**: 0.704 (explains 70% of variance)
- **MSE**: 1.067
- **MAE**: 0.874

🔑 **Top Predictors**:
1. **Years of Experience** (positive, coefficient ≈ 1.44)
2. **Age** (negative, coefficient ≈ -0.65)
3. **Education Level** (moderate effects)

## 📁 Repository Structure

```
regularized-regression-job-satisfaction/
├── data/
│   └── job_satisfaction_data.csv       # Dataset (100 observations, 7 variables)
├── notebooks/
│   └── analysis.ipynb                   # Complete Jupyter notebook with analysis
├── src/
│   ├── preprocessing.py                 # Data preprocessing utilities
│   ├── model_training.py                # Model training and evaluation
│   └── visualization.py                 # Plotting functions
├── results/
│   ├── model_comparison.png             # Performance comparison charts
│   ├── feature_importance.png           # Coefficient visualization
│   └── residual_plots.png               # Model diagnostics
├── docs/
│   └── Lab02_Analysis_Report.docx       # Detailed analysis report
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/regularized-regression-job-satisfaction.git
cd regularized-regression-job-satisfaction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Jupyter Notebook** (Interactive)
```bash
jupyter notebook notebooks/analysis.ipynb
```

**Option 2: Python Scripts** (Command Line)
```bash
# Run full analysis pipeline
python src/model_training.py
```

**Option 3: Google Colab**
- Upload `notebooks/analysis.ipynb` to [Google Colab](https://colab.research.google.com/)
- Upload `data/job_satisfaction_data.csv` when prompted
- Run all cells

## 📊 Dataset Description

**Source**: Modified employee dataset  
**Observations**: 100  
**Features**: 7

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| Gender | Categorical | Employee gender | Male, Female |
| Education_Level | Categorical | Highest education | High School, Bachelor, Master, PhD |
| Age | Numerical | Age in years | 16-70 |
| Years_of_Experience | Numerical | Work experience | -1 to 34* |
| Hours_Worked_Per_Week | Numerical | Average weekly hours | 20-62 |
| Salary | Numerical | Annual salary (USD) | $23,961 - $113,796 |
| **Job_Satisfaction** | **Target** | **Satisfaction score** | **2-14** |

\* *Note: One observation contains Years_of_Experience = -1, likely a data entry error*

## 🔬 Methodology

### 1. Data Preprocessing
- **One-hot encoding** for categorical variables (Gender, Education_Level)
- **Standardization** using StandardScaler (mean=0, std=1) - critical for regularized regression
- **Train/Validation/Test Split**: 80%/10%/10% (80/10/10 samples)

### 2. Models Evaluated

| Model | Regularization | Penalty Term | Key Characteristic |
|-------|----------------|--------------|-------------------|
| **Lasso** | L1 (Absolute) | α·Σ\|βᵢ\| | Feature selection via sparsity |
| **Ridge** | L2 (Squared) | α·Σβᵢ² | Coefficient shrinkage, handles multicollinearity |
| **Elastic Net** | L1 + L2 | α·[ρ·Σ\|βᵢ\| + (1-ρ)·Σβᵢ²] | Balanced approach |

### 3. Hyperparameter Tuning
- **Alpha values tested**: [0.01, 0.1, 1, 10, 100]
- **Selection criterion**: Lowest MSE on validation set
- **Final evaluation**: Held-out test set for unbiased performance

### 4. Evaluation Metrics
- **MSE** (Mean Squared Error) - lower is better
- **MAE** (Mean Absolute Error) - lower is better  
- **R²** (Coefficient of Determination) - higher is better (max: 1.0)
- **RMSE** (Root Mean Squared Error) - interpretable in original units

## 📈 Results Summary

### Final Test Set Performance

| Model | Alpha | MSE ↓ | MAE ↓ | R² ↑ | Rank |
|-------|-------|-------|-------|------|------|
| **Lasso** ✓ | 0.1 | **1.067** | **0.874** | **0.704** | 🥇 **1st** |
| Elastic Net | 0.1 | 1.103 | 0.882 | 0.695 | 🥈 2nd |
| Ridge | 10.0 | 1.179 | 0.889 | 0.673 | 🥉 3rd |

### Feature Coefficients (Standardized)

| Feature | Lasso | Ridge | Elastic Net | Interpretation |
|---------|-------|-------|-------------|----------------|
| Years_of_Experience | 1.441 | 1.356 | 1.409 | ⭐ Strongest positive predictor |
| Age | -0.650 | -0.690 | -0.676 | Negative relationship |
| Education_Level_Master | -0.098 | -0.152 | -0.127 | Slight negative effect |
| Gender_Male | 0.060 | 0.133 | 0.100 | Minimal effect |
| Education_Level_PhD | 0.024 | 0.108 | 0.068 | Weak positive effect |
| Hours_Worked_Per_Week | **0.000** | 0.047 | 0.010 | Eliminated by Lasso |
| Education_Level_High_School | **0.000** | -0.007 | **0.000** | Eliminated |
| Salary | **0.000** | 0.004 | **0.000** | Eliminated |

**Note**: Zero coefficients demonstrate Lasso's automatic feature selection capability.

## 🎯 Key Insights

### 1. Model Selection
- **Lasso** achieved best test performance despite Ridge leading in validation
- Demonstrates importance of held-out test set evaluation
- Low alpha (0.1) optimal for Lasso and Elastic Net
- High alpha (10.0) optimal for Ridge

### 2. Feature Importance
- **Experience dominates**: Years of experience is the strongest predictor by far
- **Age paradox**: Negative coefficient when controlling for experience suggests younger employees (at same experience level) report higher satisfaction
- **Salary surprise**: Minimal predictive power in final models - potentially mediated by other factors

### 3. Regularization Effects
- **Lasso**: Aggressive at high alpha (α ≥ 10) - severe underfitting
- **Ridge**: Monotonic improvement up to α = 10, then slight decline
- **Elastic Net**: Similar to Lasso, sensitive to alpha parameter

## 🔍 Validation vs. Test Discrepancy

**Important**: Model rankings differed between validation and test sets:
- **Validation**: Ridge > Elastic Net > Lasso
- **Test**: Lasso > Elastic Net > Ridge

This reversal highlights:
1. Small validation set (n=10) introduces high variance
2. Critical importance of final test evaluation
3. Need for cross-validation in production settings

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models and metrics
- **matplotlib** - Visualization
- **seaborn** - Statistical plotting
- **Jupyter** - Interactive development

## 📝 Limitations & Future Work

### Limitations
1. **Small sample size** (n=100) limits statistical power
2. **Tiny validation/test sets** (n=10 each) cause high variance
3. **Data quality issue**: Years_of_Experience = -1 not corrected
4. **Limited hyperparameter search**: Only 5 alpha values tested
5. **Elastic Net l1_ratio**: Only default 0.5 tested

### Recommended Improvements
- [ ] Expand dataset to n ≥ 500 observations
- [ ] Implement k-fold cross-validation
- [ ] Clean data anomalies (negative experience)
- [ ] Fine-grained hyperparameter grid search
- [ ] Test interaction terms and polynomial features
- [ ] Compare with non-linear models (Random Forest, XGBoost)
- [ ] Add confidence intervals via bootstrap
- [ ] Conduct formal residual diagnostics

## 📚 References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
3. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.

## 👤 Author

**Florencekumari Makwana**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/regularized-regression-job-satisfaction/issues).

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

**Last Updated**: February 2026
