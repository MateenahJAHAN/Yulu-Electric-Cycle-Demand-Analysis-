# Appendix A: Code Snippets Reference

This appendix contains reusable code snippets from all 5 case studies.

---

## 1. Common Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
pd.set_option('display.max_columns', None)
```

---

## 2. EDA Utility Functions

### Quick Dataset Summary

```python
def dataset_summary(df):
    """Print a comprehensive dataset summary."""
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nNumerical Stats:\n{df.describe().T}")
```

### Outlier Detection with IQR

```python
def detect_outliers_iqr(df, column, factor=1.5):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"{column}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
    print(f"  Range: [{lower:.2f}, {upper:.2f}]")
    return outliers
```

---

## 3. Confidence Interval Functions (Walmart Case Study)

### Analytical CI

```python
def analytical_ci(data, confidence=0.95):
    """Calculate analytical confidence interval."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * se
    return (mean - margin, mean + margin)
```

### Bootstrap CI

```python
def bootstrap_ci(data, n_bootstrap=2000, confidence=0.95):
    """Calculate bootstrap confidence interval."""
    data = np.array(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True))
             for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return (lower, upper)
```

### CLT Demonstration

```python
def clt_experiment(data, sample_sizes=[300, 3000, 30000], n_iterations=1000):
    """Demonstrate CLT with different sample sizes."""
    for n in sample_sizes:
        means = [np.mean(np.random.choice(data, size=n, replace=True))
                 for _ in range(n_iterations)]
        ci = (np.percentile(means, 2.5), np.percentile(means, 97.5))
        print(f"n={n:>6d} | Mean: {np.mean(means):,.2f} | "
              f"SE: {np.std(means):,.2f} | "
              f"95% CI: ({ci[0]:,.2f}, {ci[1]:,.2f})")
```

---

## 4. Hypothesis Testing Functions (Yulu Case Study)

### Two-Sample t-Test

```python
def perform_ttest(group1, group2, alpha=0.05, name1="Group1", name2="Group2"):
    """Perform and report two-sample t-test."""
    # Levene's test for equal variance
    lev_stat, lev_p = stats.levene(group1, group2)
    equal_var = lev_p > 0.05
    
    # t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    print(f"Levene's test p-value: {lev_p:.4f} ({'Equal' if equal_var else 'Unequal'} variance)")
    print(f"{name1} mean: {group1.mean():.2f}, {name2} mean: {group2.mean():.2f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to Reject H0'}")
    return t_stat, p_value
```

### One-Way ANOVA

```python
def perform_anova(*groups, alpha=0.05):
    """Perform and report one-way ANOVA."""
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.2e}")
    print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to Reject H0'}")
    return f_stat, p_value
```

### Chi-Square Test

```python
def perform_chi_square(df, col1, col2, alpha=0.05):
    """Perform and report chi-square test of independence."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"Chi-square: {chi2:.4f}, df: {dof}, p-value: {p_value:.2e}")
    print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to Reject H0'}")
    
    # Cramer's V
    n = contingency.sum().sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))
    print(f"Cramer's V: {cramers_v:.4f}")
    
    return chi2, p_value, cramers_v
```

---

## 5. Probability Functions (Aerofit Case Study)

### Marginal Probability

```python
def marginal_probability(df, column):
    """Calculate marginal probabilities for a column."""
    counts = df[column].value_counts()
    probs = counts / len(df)
    return pd.DataFrame({'Count': counts, 'Probability': probs})
```

### Conditional Probability

```python
def conditional_probability(df, target_col, condition_col):
    """Calculate P(target | condition) for all combinations."""
    ct = pd.crosstab(df[condition_col], df[target_col], normalize='index')
    return ct
```

---

## 6. Machine Learning Pipeline (LoanTap Case Study)

### Full Pipeline

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_classification_pipeline(numeric_features, categorical_features):
    """Build a complete preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
             categorical_features)
        ])
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear',
        random_state=42
    )
    
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
```

### Evaluation Report

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)

def full_evaluation(model, X_test, y_test):
    """Generate comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision: {average_precision_score(y_test, y_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return y_pred, y_prob
```
