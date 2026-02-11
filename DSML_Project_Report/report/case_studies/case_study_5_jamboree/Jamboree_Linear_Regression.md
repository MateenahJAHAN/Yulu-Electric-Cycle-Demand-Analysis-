# Chapter 7: Case Study 5 — Jamboree Education: Linear Regression

**Score Achieved: 92.0 / 100**
**Domain: Education / EdTech**
**Primary Technique: Linear Regression, Predictive Modeling, Feature Analysis**

---

## 7.1 Problem Statement

### About Jamboree Education

Jamboree Education is one of India's leading test preparation and admissions consulting companies, specializing in helping students prepare for international graduate school admissions. With a track record of guiding thousands of students into top universities worldwide, Jamboree provides coaching for standardized tests (GRE, GMAT, TOEFL, IELTS, SAT) and end-to-end application support.

### Business Problem

Jamboree Education wants to leverage data analytics to help students understand which factors most significantly influence their chances of admission to graduate programs (Master's/PhD). The company aims to:

1. **Build a predictive model** that estimates a student's probability of admission based on their academic profile
2. **Identify the most important features** that affect admission decisions
3. **Provide data-driven counseling** to students on how to optimize their application profiles
4. **Help students set realistic expectations** about their admission chances

### Objective

Build and evaluate a linear regression model to predict the chance of admission to graduate school based on academic and profile features, and provide interpretable insights for student counseling.

### Expected Outcomes

- Identify which features (GRE, TOEFL, CGPA, Research, etc.) most strongly predict admission chances
- Build a linear regression model with strong predictive performance
- Validate model assumptions through residual analysis
- Generate actionable recommendations for student profile optimization

---

## 7.2 Dataset Description

### Source and Overview

The dataset contains records of 500 student applicants with their academic profiles and corresponding admission probabilities. Each record includes standardized test scores, academic achievements, and application strength indicators.

**Dataset Dimensions:** 500 rows x 9 columns

### Schema

| Column | Data Type | Description | Value Range |
|:-------|:----------|:------------|:------------|
| `Serial No.` | Integer | Unique applicant identifier | 1 to 500 |
| `GRE Score` | Integer | Graduate Record Examination score | 260 to 340 |
| `TOEFL Score` | Integer | Test of English as a Foreign Language score | 80 to 120 |
| `University Rating` | Integer | Rating of the university applied to | 1 to 5 (1=Low, 5=Top) |
| `SOP` | Float | Statement of Purpose strength | 1.0 to 5.0 |
| `LOR` | Float | Letter of Recommendation strength | 1.0 to 5.0 |
| `CGPA` | Float | Cumulative Grade Point Average | 6.8 to 9.92 (on 10-point scale) |
| `Research` | Integer | Research experience indicator | 0 = No, 1 = Yes |
| `Chance of Admit` | Float | Probability of admission (target) | 0.34 to 0.97 |

### Feature Categories

| Category | Features | Description |
|:---------|:---------|:------------|
| **Test Scores** | GRE Score, TOEFL Score | Standardized test performance |
| **Academic** | CGPA, University Rating | Academic track record |
| **Application Strength** | SOP, LOR | Quality of application materials |
| **Experience** | Research | Research publication/experience |
| **Target** | Chance of Admit | Admission probability |

---

## 7.3 Data Loading and Initial Exploration

### Data Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("jamboree_admission.csv")

# Preview
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
```

### Structural Inspection

```python
# Data types
df.info()

# Summary statistics
df.describe()
```

### Summary Statistics

| Feature | Mean | Std | Min | Max |
|:--------|:-----|:----|:----|:----|
| GRE Score | 316.47 | 11.30 | 290 | 340 |
| TOEFL Score | 107.19 | 6.08 | 92 | 120 |
| University Rating | 3.11 | 1.14 | 1 | 5 |
| SOP | 3.37 | 0.99 | 1.0 | 5.0 |
| LOR | 3.48 | 0.93 | 1.0 | 5.0 |
| CGPA | 8.58 | 0.60 | 6.8 | 9.92 |
| Research | 0.56 | 0.50 | 0 | 1 |
| Chance of Admit | 0.72 | 0.14 | 0.34 | 0.97 |

### Key Observations

1. **No missing values** in the dataset
2. **Average admission chance** is 72% (range: 34% to 97%)
3. **GRE scores** range from 290 to 340 (out of 340), with mean at 316
4. **TOEFL scores** range from 92 to 120 (out of 120), with mean at 107
5. **56% of applicants** have research experience
6. **Average CGPA** is 8.58 out of 10, indicating strong academic applicants

---

## 7.4 Data Cleaning and Preprocessing

### Missing Value Check

```python
df.isnull().sum()
```

**Result:** No missing values found. The dataset is complete.

### Drop Irrelevant Columns

```python
# Serial number is just an identifier — drop it
df = df.drop('Serial No.', axis=1)
```

### Outlier Check

```python
# Boxplots for all numeric features
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
            'LOR', 'CGPA', 'Research', 'Chance of Admit']

for i, (ax, col) in enumerate(zip(axes.flat, features)):
    sns.boxplot(data=df, y=col, ax=ax)
    ax.set_title(f"{col}")

plt.tight_layout()
plt.show()
```

**Outlier Assessment:**
- Minor outliers detected in TOEFL Score (low-end) and CGPA (low-end)
- No extreme outliers warranting removal
- All values fall within realistic ranges for graduate applicants

### Feature Scaling Consideration

Since linear regression coefficients are sensitive to feature scales, standardization was considered but ultimately not applied to maintain interpretability of coefficients. The features are on reasonably similar scales for this dataset.

---

## 7.5 Exploratory Data Analysis

### 7.5.1 Univariate Analysis

#### Distribution of Chance of Admission (Target Variable)

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['Chance of Admit'], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Admission Chance")
plt.xlabel("Chance of Admit")
plt.ylabel("Frequency")
plt.axvline(df['Chance of Admit'].mean(), color='red', linestyle='--', label=f'Mean: {df["Chance of Admit"].mean():.2f}')
plt.legend()
plt.show()
```

**Observations:**
- The target variable is roughly **normally distributed** (good for linear regression)
- Mean admission chance: 0.72 (72%)
- Slight **left skew** — most applicants have moderate to high admission chances
- The distribution supports the use of linear regression

#### Distribution of GRE Score

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['GRE Score'], bins=25, kde=True, color='coral')
plt.title("Distribution of GRE Scores")
plt.xlabel("GRE Score")
plt.show()
```

**Observations:**
- GRE scores are approximately **normally distributed**
- Range: 290-340 with mean at 316
- Most applicants score between 305-330
- The distribution suggests a competitive applicant pool

#### Distribution of TOEFL Score

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['TOEFL Score'], bins=20, kde=True, color='green')
plt.title("Distribution of TOEFL Scores")
plt.xlabel("TOEFL Score")
plt.show()
```

**Observations:**
- TOEFL scores are approximately **normally distributed**
- Range: 92-120 with mean at 107
- Most applicants score between 100-115
- Indicates English language proficiency is generally strong in this applicant pool

#### Distribution of CGPA

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['CGPA'], bins=25, kde=True, color='purple')
plt.title("Distribution of CGPA")
plt.xlabel("CGPA (out of 10)")
plt.show()
```

**Observations:**
- CGPA follows a roughly normal distribution
- Range: 6.8-9.92 with mean at 8.58
- Most applicants have a CGPA between 8.0-9.5
- Very few applicants have CGPA below 7.5

---

### 7.5.2 Bivariate Analysis

#### Correlation Heatmap

```python
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title("Correlation Matrix")
plt.show()
```

**Correlation with Target (Chance of Admit):**

| Feature | Correlation | Strength |
|:--------|:-----------:|:---------|
| CGPA | 0.87 | Very strong positive |
| GRE Score | 0.80 | Strong positive |
| TOEFL Score | 0.79 | Strong positive |
| University Rating | 0.69 | Moderate-strong positive |
| SOP | 0.68 | Moderate-strong positive |
| LOR | 0.67 | Moderate-strong positive |
| Research | 0.55 | Moderate positive |

**Key Finding:** **CGPA is the single most important predictor** of admission chance (r = 0.87), followed by GRE Score (r = 0.80) and TOEFL Score (r = 0.79). All features show positive correlations with admission chance.

#### Scatter Plots: Key Features vs Admission Chance

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CGPA vs Admission
axes[0].scatter(df['CGPA'], df['Chance of Admit'], alpha=0.5, color='blue')
axes[0].set_xlabel('CGPA')
axes[0].set_ylabel('Chance of Admit')
axes[0].set_title('CGPA vs Admission Chance')

# GRE vs Admission
axes[1].scatter(df['GRE Score'], df['Chance of Admit'], alpha=0.5, color='red')
axes[1].set_xlabel('GRE Score')
axes[1].set_ylabel('Chance of Admit')
axes[1].set_title('GRE Score vs Admission Chance')

# TOEFL vs Admission
axes[2].scatter(df['TOEFL Score'], df['Chance of Admit'], alpha=0.5, color='green')
axes[2].set_xlabel('TOEFL Score')
axes[2].set_ylabel('Chance of Admit')
axes[2].set_title('TOEFL Score vs Admission Chance')

plt.tight_layout()
plt.show()
```

**Observations:**
- All three features show clear **linear positive relationships** with admission chance
- CGPA shows the tightest linear pattern (least scatter), confirming its highest correlation
- GRE and TOEFL show similar patterns with slightly more variability
- These linear patterns support the use of linear regression

#### Research Experience Impact

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='Research', y='Chance of Admit', data=df, palette='Set2')
plt.title("Admission Chance: Research vs No Research")
plt.xlabel("Research Experience (0=No, 1=Yes)")
plt.ylabel("Chance of Admit")
plt.show()
```

**Findings:**
- Applicants with research experience have **noticeably higher** admission chances (median ~0.77 vs ~0.65)
- The IQR for research applicants is shifted upward
- Research experience provides a clear advantage in the admissions process

---

## 7.6 Feature Engineering and Selection

### Multicollinearity Assessment (VIF Analysis)

Variance Inflation Factor (VIF) measures how much the variance of a regression coefficient is inflated due to multicollinearity.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.drop('Chance of Admit', axis=1)
X_const = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data['Feature'] = X_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)
```

**VIF Results:**

| Feature | VIF | Assessment |
|:--------|:---:|:-----------|
| const | — | N/A |
| GRE Score | 4.44 | Acceptable (< 5) |
| TOEFL Score | 3.67 | Acceptable (< 5) |
| University Rating | 2.49 | Good |
| SOP | 2.47 | Good |
| LOR | 1.95 | Good |
| CGPA | 4.30 | Acceptable (< 5) |
| Research | 1.48 | Excellent |

**Conclusion:** All VIF values are below 5 (commonly used threshold), indicating **no severe multicollinearity**. All features can be retained in the model.

### Feature Selection Decision

All 7 features are retained for the model based on:
1. All have meaningful positive correlations with the target
2. No severe multicollinearity issues (all VIF < 5)
3. Each feature represents a distinct aspect of the applicant's profile
4. Domain knowledge supports the relevance of all features

---

## 7.7 Linear Regression Modeling

### 7.7.1 Model Building

#### Train-Test Split

```python
# Define features (X) and target (y)
X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

**Split:** 400 training samples, 100 test samples

#### Model Training (Scikit-learn)

```python
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
```

#### Model Training (Statsmodels — for detailed statistics)

```python
# Using statsmodels for detailed regression summary
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())
```

### 7.7.2 Model Evaluation

#### Performance Metrics

```python
# Training set performance
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

# Test set performance
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
```

**Results:**

| Metric | Training Set | Test Set |
|:-------|:------------|:---------|
| R-squared | 0.82 | 0.80 |
| Adjusted R-squared | 0.82 | 0.78 |
| RMSE | 0.060 | 0.063 |
| MAE | 0.046 | 0.048 |

**Interpretation:**
- **R-squared = 0.82 (train) / 0.80 (test):** The model explains approximately 80-82% of the variance in admission chances. This is a strong result for a linear model.
- **Minimal overfitting:** The small gap between training and test R-squared (0.82 vs 0.80) indicates the model generalizes well to unseen data.
- **RMSE = 0.063:** On average, the model's predictions deviate by about 6.3 percentage points from actual admission chances.
- **MAE = 0.048:** The average absolute error is about 4.8 percentage points.

#### Regression Coefficients

```python
# Feature importance via coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print(coef_df)
```

**Regression Equation:**

```
Chance of Admit = -1.28
    + 0.0018 * GRE Score
    + 0.0029 * TOEFL Score
    + 0.0059 * University Rating
    + 0.0016 * SOP
    + 0.0169 * LOR
    + 0.1184 * CGPA
    + 0.0244 * Research
```

**Feature Impact Ranking (by standardized effect):**

| Rank | Feature | Coefficient | Interpretation |
|:----:|:--------|:-----------:|:---------------|
| 1 | CGPA | 0.1184 | 1-point CGPA increase = +11.8% admission chance |
| 2 | LOR | 0.0169 | 1-point LOR increase = +1.7% admission chance |
| 3 | Research | 0.0244 | Having research = +2.4% admission chance |
| 4 | University Rating | 0.0059 | 1-point rating increase = +0.6% admission chance |
| 5 | TOEFL Score | 0.0029 | 1-point TOEFL increase = +0.3% admission chance |
| 6 | GRE Score | 0.0018 | 1-point GRE increase = +0.2% admission chance |
| 7 | SOP | 0.0016 | 1-point SOP increase = +0.2% admission chance |

**Key Finding:** **CGPA is by far the most impactful feature**, with a 1-point increase in CGPA (e.g., from 8.0 to 9.0) boosting admission chances by approximately 11.8 percentage points. Research experience and LOR strength are the next most important factors.

#### Feature Importance Visualization

```python
plt.figure(figsize=(10, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue')
plt.xlabel("Coefficient Value")
plt.title("Feature Importance (Regression Coefficients)")
plt.show()
```

### 7.7.3 Residual Analysis

Residual analysis validates the assumptions of linear regression.

#### Actual vs Predicted Plot

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Admission Chance")
plt.ylabel("Predicted Admission Chance")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()
```

**Observation:** Points cluster closely around the 45-degree line, indicating good model fit. Some scatter exists at the extremes (very low and very high admission chances).

#### Residual Distribution

```python
residuals = y_test - y_pred_test

plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='coral')
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.axvline(0, color='black', linestyle='--')
plt.show()
```

**Observation:** Residuals are approximately **normally distributed** and centered around 0, satisfying the normality assumption of linear regression.

#### Q-Q Plot

```python
from scipy import stats

fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot of Residuals")
plt.show()
```

**Observation:** The Q-Q plot shows points closely following the theoretical normal line, with minor deviations at the tails. The normality assumption is reasonably well satisfied.

#### Residuals vs Predicted Values (Homoscedasticity Check)

```python
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_test, residuals, alpha=0.5, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()
```

**Observation:** Residuals appear randomly scattered around 0 without clear patterns, suggesting **homoscedasticity** (constant variance of residuals) is reasonably satisfied.

#### Durbin-Watson Test (Independence)

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(ols_model.resid)
print(f"Durbin-Watson statistic: {dw_stat:.3f}")
```

**Result:** Durbin-Watson statistic approximately 2.0 (range: 0-4, with 2 indicating no autocorrelation), confirming **independence of residuals**.

### Assumption Verification Summary

| Assumption | Test/Method | Result | Status |
|:-----------|:------------|:-------|:-------|
| Linearity | Scatter plots | Clear linear patterns | Satisfied |
| Independence | Durbin-Watson test | DW ~ 2.0 | Satisfied |
| Normality | Q-Q plot, histogram | Approximately normal | Satisfied |
| Homoscedasticity | Residual vs fitted plot | No clear patterns | Satisfied |
| No multicollinearity | VIF analysis | All VIF < 5 | Satisfied |

---

## 7.8 Key Insights and Findings

### Insight 1: CGPA is the Dominant Predictor

CGPA has the highest correlation (r = 0.87) and the largest regression coefficient. A 1-point increase in CGPA can improve admission chances by approximately 12 percentage points. This is consistent with the emphasis that top universities place on academic track record.

### Insight 2: Test Scores are Important but Secondary

While GRE (r = 0.80) and TOEFL (r = 0.79) are strongly correlated with admission, their marginal impact (per 1-point increase) is smaller than CGPA. However, combined test score improvements can have substantial effects.

### Insight 3: Research Experience Provides a Notable Boost

Having research experience adds approximately 2.4 percentage points to admission chances. For applicants with marginal profiles, this can be the differentiating factor.

### Insight 4: Application Materials (SOP, LOR) Matter

SOP and LOR have moderate correlations (~0.67-0.68) with admission. While their per-point coefficients are smaller, the qualitative nature of these components means that strong application materials can compensate for slightly lower test scores.

### Insight 5: Model Performance is Strong

The linear regression model explains ~80% of variance in admission chances (R-squared = 0.80 on test data), which is excellent for a simple linear model. The model generalizes well with minimal overfitting.

### Insight 6: All Regression Assumptions are Satisfied

The model passes all key regression diagnostic tests (linearity, normality, homoscedasticity, independence, no multicollinearity), validating the reliability of the results.

---

## 7.9 Business Recommendations

### Recommendation 1: Prioritize CGPA Improvement in Student Counseling

**Action:** Advise students early in their academic careers to focus on maintaining a high CGPA, as it has the largest impact on admission chances.

**Quantified Impact:** Moving CGPA from 8.0 to 9.0 increases admission chance by ~12 percentage points.

### Recommendation 2: Develop Balanced Preparation Strategies

**Action:** Create study plans that balance GRE/TOEFL preparation with CGPA maintenance. Students should not sacrifice academic performance for test preparation.

**Priority Order:** CGPA > GRE/TOEFL > Research > LOR/SOP

### Recommendation 3: Encourage Research Participation

**Action:** Advise students without research experience to pursue research opportunities (internships, published papers, conference presentations) before applying.

**Quantified Impact:** Research experience adds approximately 2.4 percentage points to admission probability.

### Recommendation 4: Implement Admission Chance Calculator

**Action:** Build a web-based or app-based tool using this regression model that allows students to input their scores and see their predicted admission chance, along with personalized improvement recommendations.

**Implementation:** Deploy the trained model as a simple prediction API or embedded calculator on Jamboree's website.

### Recommendation 5: Set Realistic Expectations

**Action:** Use the model to provide honest, data-backed assessments to students about their admission chances, helping them make informed decisions about which schools to target.

**Approach:** Categorize students into admission probability tiers:
- **High Chance (>80%):** Target top universities
- **Moderate Chance (60-80%):** Apply to a mix of target and safety schools
- **Low Chance (<60%):** Focus on improving profile before applying, or consider alternative programs

### Recommendation 6: Optimize University Targeting

**Action:** Use the University Rating feature to help students match their profiles with appropriately rated universities, maximizing their chances of acceptance.

---

## 7.10 Conclusion

The Jamboree Education Linear Regression case study demonstrates the complete machine learning pipeline from data exploration through model building, evaluation, and interpretation. Key achievements include:

1. **Built a linear regression model** with strong predictive performance (R-squared = 0.80 on test data)
2. **Identified CGPA as the dominant predictor** of admission chances, followed by test scores and research experience
3. **Validated all regression assumptions** through comprehensive diagnostic testing
4. **Generated actionable counseling recommendations** backed by quantitative insights
5. **Proposed a practical application** (admission chance calculator) for deploying the model

The analysis demonstrates that graduate school admission decisions can be reasonably predicted using a simple linear model with interpretable features. This provides Jamboree Education with a powerful tool for data-driven student counseling and program optimization. The analysis scored **92.0/100**, reflecting strong modeling methodology and practical business application.

### GitHub Repository

**Program:** Scaler Neovarsity DSML — Linear Regression Module
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, Google Colab

---
