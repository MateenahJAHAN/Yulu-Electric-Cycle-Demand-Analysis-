# Chapter 5: Case Study 3 — Yulu: Hypothesis Testing

**Score Achieved: 90.0 / 100**
**Domain: Micro-Mobility / Transportation**
**Primary Technique: Two-Sample t-Test, ANOVA, Chi-Square Test**

---

## 5.1 Problem Statement

### About Yulu

Yulu is India's leading micro-mobility service provider, offering shared electric cycles (e-bikes) to solve first-mile and last-mile connectivity challenges in urban areas. The company operates dockless electric bikes that commuters can pick up and drop off at designated locations. By promoting sustainable, emission-free transportation, Yulu aims to reduce traffic congestion, air pollution, and dependence on private vehicles in Indian cities.

### Business Problem

Yulu has recently experienced a significant decline in revenue. To understand the underlying causes, the management wants to determine:

1. **Which variables significantly predict the demand** for shared electric cycles?
2. **How well do these variables describe** electric cycle rental patterns?
3. **Does the working day status** affect the number of cycles rented?
4. **Do the number of cycles rented differ** across seasons and weather conditions?
5. **Is weather condition dependent on the season?**

### Objective

Apply rigorous statistical hypothesis testing to identify the significant factors influencing Yulu's electric cycle demand using a dataset of 10,886 hourly rental records.

### Expected Outcomes

- Identify which factors (season, weather, working day, temperature, humidity) significantly affect bike demand
- Quantify the strength of these relationships using appropriate statistical tests
- Generate insights to help Yulu optimize fleet distribution and improve revenue

---

## 5.2 Dataset Description

### Source and Overview

The dataset contains hourly records of bike rental counts along with associated weather, seasonal, and temporal variables. It covers a two-year period with 10,886 observations.

**Dataset Dimensions:** 10,886 rows x 12 columns

### Schema

| Column | Data Type | Description | Value Range |
|:-------|:----------|:------------|:------------|
| `datetime` | String | Date and time of record | 2011-01-01 to 2012-12-31 |
| `season` | Integer | Season code | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `holiday` | Integer | Holiday indicator | 0: Non-Holiday, 1: Holiday |
| `workingday` | Integer | Working day indicator | 0: Weekend/Holiday, 1: Working Day |
| `weather` | Integer | Weather condition | 1: Clear, 2: Mist, 3: Light Rain, 4: Heavy Rain |
| `temp` | Float | Actual temperature (Celsius) | 0.82 to 41.0 |
| `atemp` | Float | "Feels like" temperature (Celsius) | 0.76 to 45.5 |
| `humidity` | Integer | Humidity percentage | 0 to 100 |
| `windspeed` | Float | Wind speed | 0.0 to 56.99 |
| `casual` | Integer | Count of casual (non-registered) users | 0 to 367 |
| `registered` | Integer | Count of registered users | 0 to 886 |
| `count` | Integer | Total rental count (target variable) | 1 to 977 |

### Weather Condition Details

| Code | Weather Description |
|:----:|:--------------------|
| 1 | Clear, Few clouds, Partly cloudy |
| 2 | Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds |
| 3 | Light Snow, Light Rain + Thunderstorm + Scattered clouds |
| 4 | Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog |

---

## 5.3 Data Loading and Initial Exploration

### Data Import

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Load the dataset
url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/428/original/bike_sharing.csv"
df = pd.read_csv(url)

# Display first few rows
df.head()
```

### Structural Inspection

```python
# Dataset dimensions
print("Shape:", df.shape)
# Output: (10886, 12)

# Data types
df.info()

# Summary statistics
df.describe()
```

### Key Observations

| Feature | Mean | Std | Min | Max |
|:--------|:-----|:----|:----|:----|
| temp | 20.23 | 7.79 | 0.82 | 41.00 |
| atemp | 23.66 | 8.47 | 0.76 | 45.45 |
| humidity | 61.89 | 19.25 | 0.00 | 100.00 |
| windspeed | 12.80 | 8.16 | 0.00 | 56.99 |
| count | 191.57 | 181.14 | 1 | 977 |

**Observations:**
- Average bike rental count is approximately 192 per hour
- High standard deviation (181) indicates significant variation in demand
- Temperature ranges from near-freezing to hot summer conditions
- Humidity varies widely from 0% to 100%
- No missing values were found in the dataset

---

## 5.4 Data Cleaning and Type Conversion

### Missing Value Check

```python
df.isnull().sum()
```

**Result:** No missing values in any column. The dataset is clean and complete.

### Categorical Variable Conversion

Several columns are encoded as integers but are actually categorical variables. These were converted for proper analysis:

```python
# Convert categorical variables to category type
df['season'] = df['season'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weather'] = df['weather'].astype('category')
```

### Season and Weather Label Mapping

```python
# Map season labels for interpretability
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df['season_label'] = df['season'].map(season_map)

# Map weather labels
weather_map = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
df['weather_label'] = df['weather'].map(weather_map)
```

---

## 5.5 Exploratory Data Analysis

### 5.5.1 Univariate Analysis

#### Distribution of Rental Count

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['count'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Bike Rental Count")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.show()
```

**Observations:**
- The rental count distribution is **right-skewed**
- Most hourly records show between 0-200 rentals
- Peak demand periods can exceed 800-900 rentals per hour
- The skewness suggests that high-demand periods are relatively rare but significant

#### Distribution of Temperature

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['temp'], bins=30, kde=True, color='coral')
plt.title("Distribution of Temperature")
plt.xlabel("Temperature (°C)")
plt.show()
```

**Observations:**
- Temperature follows an approximately normal distribution
- Range: 0.82°C to 41.0°C with mean around 20°C
- This suggests the dataset captures all four seasons adequately

#### Distribution of Humidity

```python
plt.figure(figsize=(10, 5))
sns.histplot(df['humidity'], bins=30, kde=True, color='green')
plt.title("Distribution of Humidity")
plt.xlabel("Humidity (%)")
plt.show()
```

**Observations:**
- Humidity is slightly left-skewed with most values between 40-80%
- Some extreme values near 0% and 100% exist
- Average humidity is approximately 62%

---

### 5.5.2 Bivariate Analysis

#### Rental Count by Season

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_label', y='count', data=df,
            order=['Spring', 'Summer', 'Fall', 'Winter'],
            palette='Set2')
plt.title("Rental Count by Season")
plt.xlabel("Season")
plt.ylabel("Rental Count")
plt.show()
```

**Observations:**
- **Fall** has the highest median rental count
- **Summer** and **Winter** show moderate rental activity
- **Spring** has the lowest median rental count
- All seasons show significant outliers in the upper range
- The difference between seasons appears substantial and warrants statistical testing

#### Rental Count by Weather Condition

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather_label', y='count', data=df,
            order=['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'],
            palette='coolwarm')
plt.title("Rental Count by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Rental Count")
plt.show()
```

**Observations:**
- **Clear weather** has the highest median rental count and widest range
- **Mist/Cloudy** conditions show slightly lower but still substantial rentals
- **Light Rain/Snow** significantly reduces rental activity
- **Heavy Rain/Snow** shows the lowest rentals (very few observations in this category)
- The visual difference between weather conditions is dramatic

#### Rental Count by Working Day

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='workingday', y='count', data=df, palette='pastel')
plt.title("Rental Count: Working Day vs Non-Working Day")
plt.xlabel("Working Day (0=No, 1=Yes)")
plt.ylabel("Rental Count")
plt.show()
```

**Observations:**
- The distributions for working days and non-working days look very similar
- Medians and IQRs appear comparable
- This visual similarity suggests working day may not significantly affect rental counts

#### Temperature vs Rental Count

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='count', data=df, alpha=0.3, color='red')
plt.title("Temperature vs Rental Count")
plt.xlabel("Temperature (°C)")
plt.ylabel("Rental Count")
plt.show()
```

**Observations:**
- There is a clear **positive relationship** between temperature and rental count
- Rentals increase substantially as temperature rises from 0°C to ~35°C
- Beyond ~35°C, there may be a slight plateau or decrease (extreme heat)
- The relationship is roughly linear in the 5-35°C range

#### Correlation Heatmap

```python
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
```

**Key Correlations with `count`:**
- `temp`: 0.39 (moderate positive)
- `atemp`: 0.39 (moderate positive)
- `registered`: 0.97 (very strong positive — expected, as it's a component of count)
- `casual`: 0.69 (strong positive)
- `humidity`: -0.10 (weak negative)
- `windspeed`: 0.10 (weak positive)

---

## 5.6 Hypothesis Testing

### 5.6.1 Two-Sample t-Test: Working Day vs Rental Count

**Research Question:** Does the working day status significantly affect the number of cycles rented?

#### Hypothesis Formulation

- **Null Hypothesis (H0):** There is no significant difference in the mean rental count between working days and non-working days. (mu_working = mu_non_working)
- **Alternative Hypothesis (H1):** There is a significant difference in the mean rental count between working days and non-working days. (mu_working ≠ mu_non_working)
- **Significance Level (alpha):** 0.05

#### Test Execution

```python
from scipy import stats

# Separate data by working day status
working_day = df[df['workingday'] == 1]['count']
non_working_day = df[df['workingday'] == 0]['count']

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(working_day, non_working_day)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

#### Results

| Metric | Value |
|:-------|:------|
| t-statistic | 1.235 |
| p-value | 0.216 |
| Working Day Mean | 193.10 |
| Non-Working Day Mean | 187.67 |

#### Decision

Since **p-value (0.216) > alpha (0.05)**, we **fail to reject the null hypothesis**.

#### Interpretation

There is **no statistically significant difference** in the average number of cycles rented between working days and non-working days. The observed difference of approximately 5.4 rentals per hour is not large enough to be considered statistically meaningful.

**Business Implication:** Yulu should maintain similar fleet sizes and operational capacity on both working days and weekends/holidays. The demand pattern is relatively consistent regardless of working day status.

---

### 5.6.2 ANOVA Test: Season vs Rental Count

**Research Question:** Are the number of cycles rented significantly different across seasons?

#### Hypothesis Formulation

- **Null Hypothesis (H0):** The mean rental count is the same across all four seasons. (mu_spring = mu_summer = mu_fall = mu_winter)
- **Alternative Hypothesis (H1):** At least one season has a significantly different mean rental count.
- **Significance Level (alpha):** 0.05

#### Test Execution

```python
# Separate data by season
spring = df[df['season'] == 1]['count']
summer = df[df['season'] == 2]['count']
fall = df[df['season'] == 3]['count']
winter = df[df['season'] == 4]['count']

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(spring, summer, fall, winter)

print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.2e}")
```

#### Results

| Metric | Value |
|:-------|:------|
| F-statistic | 236.95 |
| p-value | 6.16e-149 (approximately 0) |

#### Season-wise Mean Rental Counts

| Season | Mean Rental Count | Std Dev |
|:-------|:-----------------:|:-------:|
| Spring | 111.11 | 107.53 |
| Summer | 208.34 | 176.27 |
| Fall | 236.02 | 191.03 |
| Winter | 198.87 | 181.28 |

#### Decision

Since **p-value (6.16e-149) << alpha (0.05)**, we **reject the null hypothesis**.

#### Interpretation

There is an **extremely statistically significant difference** in the average number of cycles rented across seasons. The F-statistic of 236.95 is exceptionally large, indicating that the between-group variance (across seasons) is much greater than the within-group variance.

**Key Observations:**
- **Fall** has the highest average rental count (236 per hour)
- **Spring** has the lowest average rental count (111 per hour)
- The difference between the highest (Fall) and lowest (Spring) season is more than 2x
- Summer and Winter have similar moderate rental counts

**Business Implication:** Yulu should implement **seasonal fleet management strategies**, deploying more cycles during Fall and Summer while reducing fleet size in Spring. Marketing efforts and promotional pricing should be intensified during low-demand Spring months.

---

### 5.6.3 ANOVA Test: Weather vs Rental Count

**Research Question:** Are the number of cycles rented significantly different across weather conditions?

#### Hypothesis Formulation

- **Null Hypothesis (H0):** The mean rental count is the same across all weather conditions.
- **Alternative Hypothesis (H1):** At least one weather condition has a significantly different mean rental count.
- **Significance Level (alpha):** 0.05

#### Test Execution

```python
# Separate data by weather condition
weather_1 = df[df['weather'] == 1]['count']  # Clear
weather_2 = df[df['weather'] == 2]['count']  # Mist/Cloudy
weather_3 = df[df['weather'] == 3]['count']  # Light Rain/Snow
weather_4 = df[df['weather'] == 4]['count']  # Heavy Rain/Snow (if data available)

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(weather_1, weather_2, weather_3)
# Note: Weather 4 may have very few observations

print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.2e}")
```

#### Results

| Metric | Value |
|:-------|:------|
| F-statistic | 98.28 |
| p-value | 4.97e-43 (approximately 0) |

#### Weather-wise Mean Rental Counts

| Weather | Mean Rental Count | Observation Count |
|:--------|:-----------------:|:-----------------:|
| 1 (Clear) | 205.28 | ~6,150 |
| 2 (Mist/Cloudy) | 176.57 | ~4,544 |
| 3 (Light Rain/Snow) | 111.90 | ~859 |
| 4 (Heavy Rain/Snow) | 74.33 | ~3 |

#### Decision

Since **p-value (4.97e-43) << alpha (0.05)**, we **reject the null hypothesis**.

#### Interpretation

There is a **highly significant difference** in the average number of cycles rented across weather conditions. The F-statistic of 98.28 confirms that weather has a substantial impact on bike demand.

**Key Observations:**
- **Clear weather** drives the highest demand (205 rentals/hour)
- **Misty/Cloudy** conditions reduce demand moderately (177 rentals/hour)
- **Light Rain/Snow** dramatically reduces demand (112 rentals/hour, ~45% drop from clear)
- **Heavy Rain/Snow** virtually eliminates demand (74 rentals/hour, but very few data points)
- There is a clear monotonic decrease in rentals as weather deteriorates

**Business Implication:** Yulu should implement **weather-responsive operations** — dynamically adjusting fleet availability, pricing, and notifications based on weather forecasts. During adverse weather, consider reducing operational costs and during clear weather, ensure maximum fleet availability.

---

### 5.6.4 Chi-Square Test: Weather vs Season Dependency

**Research Question:** Is the weather condition dependent on the season?

#### Hypothesis Formulation

- **Null Hypothesis (H0):** Weather and Season are independent (no relationship).
- **Alternative Hypothesis (H1):** Weather and Season are dependent (there is a relationship).
- **Significance Level (alpha):** 0.05

#### Test Execution

```python
# Create contingency table
contingency_table = pd.crosstab(df['weather'], df['season'])
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.2e}")
```

#### Results

| Metric | Value |
|:-------|:------|
| Chi-square statistic | 49.16 |
| Degrees of freedom | 9 |
| p-value | 1.55e-07 |

#### Contingency Table

|  | Spring | Summer | Fall | Winter |
|:-|:------:|:------:|:----:|:------:|
| Clear (1) | 1,471 | 1,573 | 1,570 | 1,536 |
| Mist (2) | 985 | 1,172 | 1,146 | 1,241 |
| Light Rain (3) | 271 | 207 | 199 | 182 |
| Heavy Rain (4) | 1 | 1 | 1 | 0 |

#### Decision

Since **p-value (1.55e-07) << alpha (0.05)**, we **reject the null hypothesis**.

#### Interpretation

Weather and Season are **statistically dependent** — certain weather conditions are more likely in certain seasons. The chi-square statistic of 49.16 with 9 degrees of freedom provides very strong evidence against independence.

**Key Observations:**
- **Spring** has a disproportionately higher share of Light Rain/Snow conditions (271 out of 859 total)
- **Summer** has the highest share of Mist/Cloudy conditions
- **Clear weather** is relatively evenly distributed across seasons but slightly higher in Summer and Fall
- **Heavy Rain/Snow** is extremely rare in all seasons

**Business Implication:** Since weather depends on season, Yulu can use seasonal forecasts to **proactively plan for weather-related demand changes**. For example, Spring preparations should account for higher rain probability and lower expected demand.

---

## 5.7 Key Insights and Findings

### Summary of All Hypothesis Tests

| Test | Variables | Statistic | p-value | Decision | Significance |
|:-----|:----------|:----------|:--------|:---------|:-------------|
| Two-sample t-Test | Working Day vs Count | t = 1.235 | 0.216 | Fail to reject H0 | Not Significant |
| One-way ANOVA | Season vs Count | F = 236.95 | 6.16e-149 | Reject H0 | Highly Significant |
| One-way ANOVA | Weather vs Count | F = 98.28 | 4.97e-43 | Reject H0 | Highly Significant |
| Chi-square | Weather vs Season | chi2 = 49.16 | 1.55e-07 | Reject H0 | Highly Significant |

### Key Takeaways

1. **Season is the strongest predictor** of bike rental demand (F = 236.95), with Fall seeing 2x the demand of Spring
2. **Weather has a dramatic impact** on demand (F = 98.28), with clear weather driving ~84% more rentals than rainy conditions
3. **Working day does not matter** — demand is similar on weekdays and weekends, suggesting both commuters and recreational riders contribute to the user base
4. **Weather and Season are related**, enabling predictive planning based on seasonal weather expectations
5. **Temperature has a moderate positive correlation** (0.39) with demand, consistent with the season and weather findings

---

## 5.8 Business Recommendations

### Recommendation 1: Implement Seasonal Fleet Management

**Action:** Deploy 40-50% more bikes during Fall and Summer compared to Spring. Create a seasonal capacity planning calendar.

**Rationale:** Fall demand is 2.1x Spring demand, requiring proportional fleet adjustments.

### Recommendation 2: Weather-Based Dynamic Operations

**Action:** Integrate real-time weather data into the operational dashboard. Reduce active fleet during adverse weather and maximize availability during clear conditions.

**Rationale:** Clear weather drives 84% more rentals than light rain; weather-based fleet optimization can reduce operational costs while maintaining service quality.

### Recommendation 3: Consistent Weekday/Weekend Operations

**Action:** Maintain uniform fleet sizes and operational capacity across weekdays and weekends, as demand does not significantly differ.

**Rationale:** The t-test (p = 0.216) confirms no significant difference between working day and non-working day demand.

### Recommendation 4: Temperature-Responsive Marketing

**Action:** Increase marketing efforts (push notifications, email campaigns) on warm, clear days when demand potential is highest.

**Rationale:** The positive correlation between temperature and demand (r = 0.39) combined with weather effects creates predictable high-demand conditions.

### Recommendation 5: Spring Revenue Recovery Strategy

**Action:** Implement promotional pricing, loyalty bonuses, and user engagement campaigns specifically during Spring to mitigate the seasonal demand trough.

**Rationale:** Spring has the lowest average demand (111 rentals/hour vs 236 for Fall), representing the biggest opportunity for improvement.

### Recommendation 6: Predictive Demand Forecasting

**Action:** Build a demand forecasting model incorporating season, weather, temperature, and humidity as key features to predict hourly demand 24-72 hours in advance.

**Rationale:** The statistical analysis confirms these variables significantly predict demand and can be incorporated into machine learning models.

---

## 5.9 Conclusion

The Yulu Hypothesis Testing case study demonstrates the rigorous application of statistical tests to identify factors influencing micro-mobility demand. Through a systematic approach using t-tests, ANOVA, and chi-square tests, the analysis conclusively established that:

1. **Season and weather are the dominant factors** affecting electric cycle demand
2. **Working day status is irrelevant** to demand patterns
3. **Weather conditions depend on season**, enabling seasonal forecasting approaches
4. **Temperature has a moderate positive influence** on rental activity

These statistically validated findings provide a strong foundation for Yulu to optimize operations, reduce costs during low-demand periods, and maximize revenue during favorable conditions. The analysis scored **90.0/100**, demonstrating solid hypothesis testing methodology and practical business application.

### GitHub Repository

**Repository:** [Yulu-Electric-Cycle-Demand-Analysis-](https://github.com/MateenahJAHAN/Yulu-Electric-Cycle-Demand-Analysis-)
**Notebook:** `Yulu.ipynb`
**Dataset:** `bike_sharing.csv`
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scipy.stats, Google Colab

---
