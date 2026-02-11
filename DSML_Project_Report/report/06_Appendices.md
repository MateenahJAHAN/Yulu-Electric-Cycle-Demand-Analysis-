# Appendices

---

# Appendix A: Python Environment Setup

## A.1 Required Libraries

To reproduce the analyses in this report, install the following Python libraries:

```python
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
```

## A.2 Installation

### Using pip

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### Using conda

```bash
conda install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### Using Google Colab

Google Colab comes with all required libraries pre-installed. Simply open a new notebook and begin coding.

## A.3 Standard Import Block

The following import block was used across all case studies:

```python
# Standard imports for all case studies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette('Set2')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# For inline plots in Jupyter
%matplotlib inline
```

---

# Appendix B: Key Code Snippets

## B.1 Data Quality Assessment Template

```python
def data_quality_report(df):
    """Generate a comprehensive data quality report."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    if missing.sum() == 0:
        print("No missing values found!")
    
    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    # Duplicates
    dupes = df.duplicated().sum()
    print(f"\n--- Duplicates: {dupes} ---")
    
    # Numeric summary
    print("\n--- Numeric Summary ---")
    print(df.describe().round(2))
    
    return None

# Usage: data_quality_report(df)
```

## B.2 Confidence Interval Functions

```python
def analytic_ci(data, confidence=0.95):
    """
    Calculate confidence interval using the normal approximation.
    
    Parameters:
    -----------
    data : array-like
        Sample data
    confidence : float
        Confidence level (default 0.95)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence, 1.96)
    
    margin_of_error = z * (std / np.sqrt(n))
    return (mean - margin_of_error, mean + margin_of_error)


def bootstrap_ci(data, n_bootstrap=2000, confidence=0.95):
    """
    Calculate bootstrap confidence interval.
    
    Parameters:
    -----------
    data : array-like
        Sample data
    n_bootstrap : int
        Number of bootstrap samples (default 2000)
    confidence : float
        Confidence level (default 0.95)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return (lower, upper)
```

## B.3 Hypothesis Testing Functions

```python
from scipy import stats

def perform_ttest(group1, group2, group1_name="Group 1", group2_name="Group 2", alpha=0.05):
    """
    Perform and report a two-sample t-test.
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    print(f"\n{'='*50}")
    print(f"TWO-SAMPLE T-TEST: {group1_name} vs {group2_name}")
    print(f"{'='*50}")
    print(f"Mean {group1_name}: {np.mean(group1):.2f}")
    print(f"Mean {group2_name}: {np.mean(group2):.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Alpha: {alpha}")
    
    if p_value < alpha:
        print(f"\nResult: REJECT H0 (p={p_value:.6f} < alpha={alpha})")
        print("There IS a statistically significant difference.")
    else:
        print(f"\nResult: FAIL TO REJECT H0 (p={p_value:.6f} > alpha={alpha})")
        print("There is NO statistically significant difference.")
    
    return t_stat, p_value


def perform_anova(*groups, group_names=None, alpha=0.05):
    """
    Perform and report a one-way ANOVA test.
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n{'='*50}")
    print("ONE-WAY ANOVA TEST")
    print(f"{'='*50}")
    
    if group_names:
        for name, group in zip(group_names, groups):
            print(f"Mean {name}: {np.mean(group):.2f} (n={len(group)})")
    
    print(f"\nF-statistic: {f_stat:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Alpha: {alpha}")
    
    if p_value < alpha:
        print(f"\nResult: REJECT H0 (p={p_value:.2e} < alpha={alpha})")
        print("At least one group mean is significantly different.")
    else:
        print(f"\nResult: FAIL TO REJECT H0 (p={p_value:.2e} > alpha={alpha})")
        print("No significant difference between group means.")
    
    return f_stat, p_value


def perform_chi_square(df, var1, var2, alpha=0.05):
    """
    Perform and report a chi-square test of independence.
    """
    contingency = pd.crosstab(df[var1], df[var2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\n{'='*50}")
    print(f"CHI-SQUARE TEST: {var1} vs {var2}")
    print(f"{'='*50}")
    print(f"\nContingency Table:")
    print(contingency)
    print(f"\nChi-square statistic: {chi2:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.2e}")
    print(f"Alpha: {alpha}")
    
    if p_value < alpha:
        print(f"\nResult: REJECT H0 (p={p_value:.2e} < alpha={alpha})")
        print(f"{var1} and {var2} are DEPENDENT (related).")
    else:
        print(f"\nResult: FAIL TO REJECT H0 (p={p_value:.2e} > alpha={alpha})")
        print(f"{var1} and {var2} are INDEPENDENT (not related).")
    
    return chi2, p_value, dof
```

## B.4 Linear Regression Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def regression_pipeline(X, y, test_size=0.2, random_state=42):
    """
    Complete linear regression pipeline with evaluation.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scikit-learn model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    print("=" * 50)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 50)
    
    print("\n--- Training Set ---")
    print(f"R-squared: {r2_score(y_train, y_pred_train):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    
    print("\n--- Test Set ---")
    print(f"R-squared: {r2_score(y_test, y_pred_test):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # Coefficients
    print("\n--- Coefficients ---")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    print(coef_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:.4f}")
    
    # VIF
    print("\n--- VIF Analysis ---")
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X_const.values, i+1) 
                for i in range(X.shape[1])]
    })
    print(vif_data.to_string(index=False))
    
    return model, X_test, y_test, y_pred_test
```

## B.5 EDA Visualization Template

```python
def comprehensive_eda(df, target_col=None):
    """
    Generate comprehensive EDA visualizations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Distribution of numeric features
    n_numeric = len(numeric_cols)
    if n_numeric > 0:
        fig, axes = plt.subplots(
            (n_numeric + 2) // 3, 3, 
            figsize=(15, 4 * ((n_numeric + 2) // 3))
        )
        axes = axes.flatten() if n_numeric > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(f'Distribution: {col}')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', alpha=0.7)
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle("Univariate Analysis — Numeric Features", y=1.02, fontsize=14)
        plt.show()
    
    # 2. Categorical features
    if categorical_cols:
        fig, axes = plt.subplots(
            1, min(len(categorical_cols), 3),
            figsize=(5 * min(len(categorical_cols), 3), 4)
        )
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols[:3]):
            sns.countplot(data=df, x=col, ax=axes[i], palette='Set2')
            axes[i].set_title(f'Distribution: {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # 3. Correlation heatmap
    if n_numeric > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                    mask=mask, center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    
    # 4. Boxplots if target is specified
    if target_col and categorical_cols:
        for col in categorical_cols[:3]:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x=col, y=target_col, palette='Set2')
            plt.title(f'{target_col} by {col}')
            plt.xticks(rotation=45)
            plt.show()
```

---

# Appendix C: Statistical Formulas Reference

## C.1 Descriptive Statistics

### Mean (Arithmetic Average)
```
x_bar = (1/n) * sum(x_i) for i = 1 to n
```

### Variance
```
s^2 = (1/(n-1)) * sum((x_i - x_bar)^2) for i = 1 to n
```

### Standard Deviation
```
s = sqrt(s^2)
```

### Standard Error of the Mean
```
SE = s / sqrt(n)
```

### Interquartile Range
```
IQR = Q3 - Q1
```

### Outlier Bounds (IQR Method)
```
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

---

## C.2 Probability

### Marginal Probability
```
P(A) = n(A) / n(Total)
```

### Conditional Probability
```
P(A|B) = P(A and B) / P(B)
```

### Bayes' Theorem
```
P(A|B) = P(B|A) * P(A) / P(B)
```

---

## C.3 Confidence Intervals

### For Population Mean (Large Sample)
```
CI = x_bar +/- z_(alpha/2) * (s / sqrt(n))
```

### Z-scores for Common Confidence Levels
```
90% CI: z = 1.645
95% CI: z = 1.96
99% CI: z = 2.576
```

### Margin of Error
```
MOE = z_(alpha/2) * (s / sqrt(n))
```

---

## C.4 Hypothesis Testing

### t-Test Statistic (Two Independent Samples)
```
t = (x_bar_1 - x_bar_2) / sqrt(s_1^2/n_1 + s_2^2/n_2)
```

### F-Statistic (ANOVA)
```
F = MS_between / MS_within

where:
MS_between = SS_between / (k - 1)
MS_within = SS_within / (N - k)
k = number of groups
N = total observations
```

### Chi-Square Statistic
```
chi^2 = sum((O_i - E_i)^2 / E_i) for all cells

where:
O_i = observed frequency
E_i = expected frequency = (row_total * column_total) / grand_total
```

### Degrees of Freedom (Chi-Square)
```
df = (r - 1) * (c - 1)

where:
r = number of rows
c = number of columns
```

---

## C.5 Linear Regression

### Simple Linear Regression
```
y = beta_0 + beta_1 * x + epsilon
```

### Multiple Linear Regression
```
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_p*x_p + epsilon
```

### R-squared (Coefficient of Determination)
```
R^2 = 1 - (SS_res / SS_tot)

where:
SS_res = sum((y_i - y_hat_i)^2)  [Residual sum of squares]
SS_tot = sum((y_i - y_bar)^2)    [Total sum of squares]
```

### Adjusted R-squared
```
R^2_adj = 1 - ((1 - R^2) * (n - 1) / (n - p - 1))

where:
n = number of observations
p = number of predictors
```

### RMSE (Root Mean Squared Error)
```
RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
```

### MAE (Mean Absolute Error)
```
MAE = (1/n) * sum(|y_i - y_hat_i|)
```

### VIF (Variance Inflation Factor)
```
VIF_j = 1 / (1 - R^2_j)

where:
R^2_j = R-squared from regressing x_j on all other predictors
VIF > 5 indicates potential multicollinearity
VIF > 10 indicates serious multicollinearity
```

### Durbin-Watson Statistic
```
DW = sum((e_t - e_{t-1})^2) / sum(e_t^2) for t = 2 to n

DW ~ 2: No autocorrelation
DW < 2: Positive autocorrelation
DW > 2: Negative autocorrelation
```

---

# Appendix D: Glossary of Terms

| Term | Definition |
|:-----|:-----------|
| **ANOVA** | Analysis of Variance — statistical test comparing means of 3+ groups |
| **Bivariate Analysis** | Analysis examining the relationship between two variables |
| **Bootstrap** | Resampling technique to estimate statistics by drawing samples with replacement |
| **Central Limit Theorem** | Theorem stating sampling distribution of mean approaches normal as sample size increases |
| **Chi-Square Test** | Non-parametric test of independence between two categorical variables |
| **CLT** | Central Limit Theorem |
| **Confidence Interval** | Range of values likely to contain the true population parameter |
| **Conditional Probability** | Probability of an event given that another event has occurred |
| **Correlation** | Statistical measure of the strength and direction of linear relationship between two variables |
| **CRISP-DM** | Cross-Industry Standard Process for Data Mining |
| **CGPA** | Cumulative Grade Point Average |
| **Degrees of Freedom** | Number of independent values that can vary in a statistical calculation |
| **Descriptive Statistics** | Statistics that summarize and describe features of a dataset |
| **Durbin-Watson** | Test statistic for detecting autocorrelation in regression residuals |
| **EDA** | Exploratory Data Analysis |
| **F-statistic** | Test statistic for ANOVA, ratio of between-group to within-group variance |
| **Feature Engineering** | Process of creating new features from existing data |
| **Heteroscedasticity** | Unequal variance of residuals across values of a predictor |
| **Homoscedasticity** | Equal variance of residuals across values of a predictor |
| **Hypothesis Testing** | Statistical method for making decisions based on data |
| **IQR** | Interquartile Range (Q3 - Q1) |
| **KDE** | Kernel Density Estimation — non-parametric way to estimate probability density |
| **Linear Regression** | Statistical method modeling linear relationship between dependent and independent variables |
| **MAE** | Mean Absolute Error |
| **Marginal Probability** | Probability of an event regardless of outcomes of other events |
| **Multicollinearity** | High correlation among predictor variables in regression |
| **Normality** | Assumption that data follows a normal (Gaussian) distribution |
| **Null Hypothesis (H0)** | Default assumption in hypothesis testing (typically "no effect" or "no difference") |
| **p-value** | Probability of observing data as extreme as the sample, assuming H0 is true |
| **Q-Q Plot** | Quantile-Quantile plot for assessing normality of residuals |
| **R-squared** | Proportion of variance in the dependent variable explained by the model |
| **Residual** | Difference between observed and predicted values |
| **RMSE** | Root Mean Squared Error |
| **Significance Level (alpha)** | Threshold for rejecting the null hypothesis (typically 0.05) |
| **Skewness** | Measure of asymmetry in a distribution |
| **t-Test** | Statistical test comparing means of two groups |
| **Univariate Analysis** | Analysis examining one variable at a time |
| **VIF** | Variance Inflation Factor — measure of multicollinearity |

---

**END OF REPORT**

---

*Report compiled by Mateenah Jahan*
*Scaler Neovarsity — Data Science & Machine Learning Program*
*February 2026*

---
