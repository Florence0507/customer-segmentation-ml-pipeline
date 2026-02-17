# Data Directory

## Dataset: Job Satisfaction Data

**File**: `job_satisfaction_data.csv`

### Description

Employee dataset containing demographic information, work characteristics, and job satisfaction scores.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| Gender | Categorical | Employee gender (Male, Female) |
| Education_Level | Categorical | Highest education (High School, Bachelor, Master, PhD) |
| Age | Numerical | Age in years (range: 16-70) |
| Years_of_Experience | Numerical | Professional experience in years (range: -1 to 34)* |
| Hours_Worked_Per_Week | Numerical | Average weekly work hours (range: 20-62) |
| Salary | Numerical | Annual salary in USD (range: $23,961-$113,796) |
| **Job_Satisfaction** | **Target** | **Satisfaction score (range: 2-14)** |

\* *Note: Contains one anomalous value (-1) representing a data quality issue*

### Statistics

- **Observations**: 100
- **Features**: 7 (5 numerical, 2 categorical)
- **Target**: Job_Satisfaction (continuous)
- **Missing Values**: None
- **Class Distribution**: N/A (regression task)

### Data Quality Notes

1. **Negative Experience**: One observation has `Years_of_Experience = -1`
   - Likely data entry error
   - Retained for analysis due to small dataset size
   - Should be investigated/corrected in production

2. **Age-Experience Relationship**: Some unusual combinations exist
   - Example: Age 16 with 29 years experience
   - Suggests data collection inconsistencies

### Usage

```python
import pandas as pd

# Load data
df = pd.read_csv('data/job_satisfaction_data.csv')

# Basic exploration
print(df.shape)
print(df.info())
print(df.describe())
```

### Citation

If using this dataset, please cite:

```
Makwana, F. (2026). Job Satisfaction Analysis Dataset. 
GitHub: https://github.com/YOUR-USERNAME/regularized-regression-job-satisfaction
```

### License

This dataset is available under the MIT License (see LICENSE file).
