# Appendix B: Statistical Tables and Formulas

---

## 1. Descriptive Statistics Formulas

### Measures of Central Tendency

| Measure | Formula | Description |
|---------|---------|-------------|
| **Mean** | x_bar = (1/n) * sum(x_i) | Average of all values |
| **Median** | Middle value when sorted | 50th percentile |
| **Mode** | Most frequent value | Most common observation |

### Measures of Dispersion

| Measure | Formula | Description |
|---------|---------|-------------|
| **Range** | max - min | Spread of data |
| **Variance** | s^2 = (1/(n-1)) * sum((x_i - x_bar)^2) | Average squared deviation |
| **Std Dev** | s = sqrt(variance) | Spread in original units |
| **IQR** | Q3 - Q1 | Middle 50% spread |
| **CV** | (s / x_bar) * 100% | Relative variability |

### Measures of Shape

| Measure | Interpretation |
|---------|----------------|
| **Skewness = 0** | Symmetric distribution |
| **Skewness > 0** | Right-skewed (tail extends right) |
| **Skewness < 0** | Left-skewed (tail extends left) |
| **Kurtosis = 3** | Normal (mesokurtic) |
| **Kurtosis > 3** | Heavy-tailed (leptokurtic) |
| **Kurtosis < 3** | Light-tailed (platykurtic) |

---

## 2. Probability Formulas

| Concept | Formula | Description |
|---------|---------|-------------|
| **Marginal** | P(A) = n(A) / n(S) | Probability of single event |
| **Joint** | P(A and B) = n(A and B) / n(S) | Probability of both events |
| **Conditional** | P(A|B) = P(A and B) / P(B) | Probability given another event |
| **Bayes' Theorem** | P(A|B) = P(B|A) * P(A) / P(B) | Inverse conditional probability |

---

## 3. Confidence Interval Formulas

### For Population Mean (known sigma)

CI = x_bar +/- z_(alpha/2) * (sigma / sqrt(n))

### For Population Mean (unknown sigma)

CI = x_bar +/- t_(alpha/2, n-1) * (s / sqrt(n))

### Critical z-Values

| Confidence Level | Alpha | z-value |
|-----------------|-------|---------|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 98% | 0.02 | 2.326 |
| 99% | 0.01 | 2.576 |

---

## 4. Hypothesis Testing Reference

### General Procedure

1. State H0 (null) and H1 (alternative)
2. Choose significance level (alpha, usually 0.05)
3. Select appropriate test statistic
4. Calculate test statistic and p-value
5. Compare p-value with alpha
6. Make decision: Reject H0 if p-value < alpha

### Test Selection Guide

| Data Type | Groups | Test |
|-----------|--------|------|
| Continuous | 2 groups | Two-sample t-test |
| Continuous | 3+ groups | One-way ANOVA |
| Categorical | 2 variables | Chi-square test |
| Continuous | Before/After | Paired t-test |
| Continuous | Non-normal | Mann-Whitney U test |

### Two-Sample t-Test

t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

**Assumptions:**
- Independent samples
- Approximately normal (or large n via CLT)
- Equal variance (check with Levene's test)

### One-Way ANOVA

F = MS_between / MS_within

Where:
- MS_between = SS_between / (k - 1)
- MS_within = SS_within / (N - k)
- k = number of groups
- N = total observations

**Assumptions:**
- Independent observations
- Normal distribution within groups
- Equal variances (homoscedasticity)

### Chi-Square Test of Independence

chi^2 = sum((O_i - E_i)^2 / E_i)

Where:
- O_i = observed frequency
- E_i = expected frequency = (row_total * col_total) / grand_total
- df = (rows - 1) * (cols - 1)

**Effect Size: Cramer's V**

V = sqrt(chi^2 / (n * min(r-1, c-1)))

| V | Interpretation |
|---|----------------|
| < 0.1 | Negligible |
| 0.1 - 0.3 | Weak |
| 0.3 - 0.5 | Moderate |
| > 0.5 | Strong |

---

## 5. Machine Learning Metrics

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP / (TP+FP) | Cost of FP is high |
| **Recall** | TP / (TP+FN) | Cost of FN is high |
| **F1 Score** | 2*P*R / (P+R) | Balance P and R |
| **Specificity** | TN / (TN+FP) | True negative rate |
| **ROC AUC** | Area under ROC curve | Overall discrimination |
| **PR AUC** | Area under PR curve | Imbalanced classes |

### Confusion Matrix Layout

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

---

## 6. Summary of All Statistical Tests Performed

### Case Study 3: Yulu

| Test | H0 | H1 | Statistic | p-value | Decision |
|------|----|----|-----------|---------|----------|
| t-test | mu_working = mu_nonworking | mu_working != mu_nonworking | t=1.24 | 0.216 | Fail to Reject |
| ANOVA (Season) | All season means equal | At least one differs | F=236.95 | 6.16e-149 | Reject |
| ANOVA (Weather) | All weather means equal | At least one differs | F=98.28 | 4.97e-43 | Reject |
| Chi-Square | Weather independent of Season | Weather depends on Season | chi2=49.16 | 1.55e-07 | Reject |

### Case Study 2: Walmart

| Comparison | CI Lower | CI Upper | Overlap | Significant |
|-----------|----------|----------|---------|-------------|
| Male spending | 9,422 | 9,453 | No overlap with Female | Yes |
| Female spending | 8,710 | 8,759 | No overlap with Male | Yes |
| Married spending | 9,240 | 9,282 | Overlaps with Unmarried | No |
| Unmarried spending | 9,249 | 9,283 | Overlaps with Married | No |

### Case Study 5: LoanTap

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC AUC | 0.574 | Modest discrimination |
| PR AUC | 0.108 | Challenged by imbalance |
| Accuracy | 0.783 | Misleading with imbalance |
| Precision (default) | 0.114 | Low - many false alarms |
| Recall (default) | 0.245 | Catches ~25% of defaulters |
| F1 (default) | 0.156 | Low overall performance |
