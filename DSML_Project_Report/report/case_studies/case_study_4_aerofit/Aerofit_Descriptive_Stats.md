# Chapter 6: Case Study 4 — Aerofit: Descriptive Statistics & Probability

**Score Achieved: 86.0 / 100**
**Domain: Fitness Equipment / Retail**
**Primary Technique: Descriptive Statistics, Probability Analysis, Customer Profiling**

---

## 6.1 Problem Statement

### About Aerofit

Aerofit is a leading brand in the fitness equipment industry, specializing in treadmills. The company offers three distinct treadmill products, each targeting different customer segments:

| Product | Level | Price | Key Features |
|:--------|:------|:------|:-------------|
| **KP281** | Entry-level | $1,500 | Basic features, suitable for beginners |
| **KP481** | Mid-level | $1,750 | Enhanced features, better build quality |
| **KP781** | Advanced | $2,500 | Premium features, high performance, professional grade |

### Business Problem

The Aerofit business team wants to understand:

1. **Who are the typical customers** buying each treadmill model?
2. **What customer characteristics** (age, gender, income, fitness level) are associated with each product?
3. **What is the probability** that a customer with certain characteristics will purchase a specific treadmill?
4. How can these insights be used to create **targeted marketing strategies** for each product tier?

### Objective

Analyze the customer dataset using descriptive statistics, probability calculations, and customer profiling techniques to build comprehensive buyer personas for each treadmill model and generate targeted marketing recommendations.

### Expected Outcomes

- Describe customer demographics using statistical summaries
- Detect and handle outliers appropriately
- Calculate marginal and conditional probabilities for product purchases
- Build detailed customer profiles for each treadmill tier
- Provide actionable marketing recommendations

---

## 6.2 Dataset Description

### Source and Overview

The dataset contains information about 180 customers who purchased Aerofit treadmills in the last 3 months. Each record captures the customer's demographic and fitness profile along with the treadmill model purchased.

**Dataset Dimensions:** 180 rows x 9 columns

### Schema

| Column | Data Type | Description | Example Values |
|:-------|:----------|:------------|:---------------|
| `Product` | String | Treadmill model purchased | KP281, KP481, KP781 |
| `Age` | Integer | Customer age in years | 18, 25, 35, 45 |
| `Gender` | String | Customer gender | Male, Female |
| `Education` | Integer | Years of education | 14, 16, 18, 21 |
| `MaritalStatus` | String | Marital status | Single, Partnered |
| `Usage` | Integer | Planned treadmill usage (times/week) | 2, 3, 4, 5, 7 |
| `Fitness` | Integer | Self-rated fitness level (1-5) | 1 (Poor) to 5 (Excellent) |
| `Income` | Integer | Annual income in USD | $29,562 to $104,581 |
| `Miles` | Integer | Expected miles to walk/run per week | 21 to 360 |

---

## 6.3 Data Loading and Initial Exploration

### Data Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("aerofit_treadmill.csv")

# Preview
print("Preview of dataset:")
print(df.head())

# Shape
print("\nShape of dataset (rows, columns):", df.shape)
# Output: (180, 9)

# Data types
print("\nData types:")
print(df.dtypes)
```

### Summary Statistics

```python
df.describe()
```

| Statistic | Age | Education | Usage | Fitness | Income | Miles |
|:----------|----:|----------:|------:|--------:|-------:|------:|
| Count | 180 | 180 | 180 | 180 | 180 | 180 |
| Mean | 28.79 | 15.57 | 3.46 | 3.31 | $53,720 | 103.19 |
| Std | 6.94 | 1.62 | 1.08 | 0.96 | $16,896 | 51.86 |
| Min | 18 | 12 | 2 | 1 | $29,562 | 21 |
| 25% | 24 | 14 | 3 | 3 | $44,058 | 66 |
| 50% | 26 | 16 | 3 | 3 | $50,597 | 94 |
| 75% | 33 | 16 | 4 | 4 | $58,668 | 115 |
| Max | 50 | 21 | 7 | 5 | $104,581 | 360 |

### Key Observations from Initial Exploration

1. **No missing values** were found in any column
2. **Average customer** is approximately 29 years old with 16 years of education
3. **Average income** is approximately $53,720 per year
4. **Average fitness level** is 3.3 out of 5 (moderate)
5. **Treadmill usage** averages 3-4 times per week
6. **Expected weekly miles** range from 21 to 360, averaging ~103 miles

---

## 6.4 Outlier Detection

### Statistical Approach

```python
print("Summary statistics:")
print(df.describe())
```

**Comparing Mean vs Median (50th percentile):**

| Feature | Mean | Median | Skewness Indicator |
|:--------|:-----|:-------|:-------------------|
| Age | 28.79 | 26 | Right-skewed (older outliers) |
| Income | $53,720 | $50,597 | Right-skewed (high earners) |
| Miles | 103.19 | 94 | Right-skewed (heavy runners) |
| Fitness | 3.31 | 3 | Slightly right-skewed |

### Visual Approach — Boxplots

```python
num_cols = ["Age", "Education", "Usage", "Fitness", "Income", "Miles"]

plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()
```

### Outlier Assessment

| Feature | Outliers Detected | Assessment |
|:--------|:-----------------|:-----------|
| Age | Few (ages 45-50) | Legitimate — some older fitness enthusiasts |
| Education | Few (21 years) | Legitimate — doctoral-level education |
| Usage | Few (7 times/week) | Legitimate — daily users |
| Fitness | None significant | Normal range |
| Income | Several ($85K-$105K) | Legitimate — high-income professionals |
| Miles | Several (200-360 miles) | Legitimate — serious runners/athletes |

**Decision:** All outliers appear to represent real extreme customers (e.g., athletes, high-income professionals) rather than data errors. **No outliers were removed** as they provide valuable information about premium customer segments.

---

## 6.5 Exploratory Data Analysis

### 6.5.1 Univariate Analysis

#### Distribution of Products Purchased

```python
plt.figure(figsize=(6, 4))
sns.countplot(x='Product', data=df)
plt.title("Distribution of Products Purchased")
plt.show()
```

**Product Distribution:**

| Product | Count | Percentage |
|:--------|------:|-----------:|
| KP281 | 80 | 44.4% |
| KP481 | 60 | 33.3% |
| KP781 | 40 | 22.2% |

**Observation:** The entry-level KP281 is the most popular product, capturing nearly half of all sales. As price increases, the customer count decreases — typical for a tiered product line.

#### Gender Distribution

```python
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()
```

**Findings:**
- **Males** form the majority of treadmill buyers
- **Females** represent a significant but smaller share
- This gender split varies significantly by product tier (explored in bivariate analysis)

#### Age Distribution

```python
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution of Customers")
plt.show()
```

**Findings:**
- Most customers are between **20-35 years old**
- Peak age group is **mid-twenties**
- A secondary cluster exists around 35-40 years
- Very few customers below 20 or above 45

#### Income Distribution

```python
plt.figure(figsize=(8, 5))
sns.histplot(df['Income'], bins=20, kde=True, color='green')
plt.title("Income Distribution")
plt.show()
```

**Findings:**
- Income is **right-skewed** with most customers earning $40K-$60K
- A tail of high-income customers extends to $100K+
- The median income ($50,597) is lower than the mean ($53,720), confirming the right skew

#### Fitness Level Distribution

```python
plt.figure(figsize=(6, 4))
sns.countplot(x='Fitness', data=df)
plt.title("Self-Rated Fitness Level Distribution")
plt.show()
```

**Findings:**
- Most customers rate themselves at **fitness level 3** (moderate)
- Levels 4 and 5 (fit/very fit) are less common but represent the premium segment
- Very few customers rate themselves at level 1 (poor fitness)

---

### 6.5.2 Bivariate Analysis

#### Product vs Gender

```python
plt.figure(figsize=(6, 4))
sns.countplot(x="Product", hue="Gender", data=df)
plt.title("Product vs Gender")
plt.show()
```

**Findings:**
- Both males and females buy all products, but patterns differ
- **KP281:** Relatively balanced gender split
- **KP481:** Slight male majority
- **KP781:** Strong male dominance — males are 3x more likely to purchase the advanced treadmill

#### Product vs Marital Status

```python
plt.figure(figsize=(6, 4))
sns.countplot(x="Product", hue="MaritalStatus", data=df)
plt.title("Product vs Marital Status")
plt.show()
```

**Findings:**
- **Partnered** people tend to buy mid and high-end models more than singles
- **Singles** favor the entry-level KP281
- Marital status shows some influence on product choice but is not as strong as gender

#### Product vs Age (Boxplot)

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Product", y="Age", data=df)
plt.title("Product vs Age")
plt.show()
```

**Findings:**
- **KP281 buyers** tend to be younger (median ~24-25 years)
- **KP481 buyers** are in the middle age range (median ~26-28 years)
- **KP781 buyers** tend to be older (median ~28-30 years)
- Clear age progression from entry-level to advanced

#### Product vs Income (Boxplot)

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Product", y="Income", data=df)
plt.title("Product vs Income")
plt.show()
```

**Findings:**
- Strong positive relationship between income and product tier
- **KP281 buyers:** Median income ~$40K-$45K
- **KP481 buyers:** Median income ~$50K-$55K
- **KP781 buyers:** Median income ~$65K-$75K
- Income is a clear differentiator for product selection

#### Product vs Fitness Level (Boxplot)

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Product", y="Fitness", data=df)
plt.title("Product vs Self-Rated Fitness Level")
plt.show()
```

**Findings:**
- **KP281 buyers** have lower fitness levels (median = 2-3)
- **KP781 buyers** have higher fitness levels (median = 4-5)
- Fitness level strongly predicts product choice — fitter people invest in better equipment

---

## 6.6 Probability Analysis

### 6.6.1 Marginal Probabilities

**Question:** What is the probability that a randomly selected customer purchased each product?

```python
# Marginal Probability - product distribution
product_counts = df['Product'].value_counts(normalize=True) * 100
print("Marginal Probability (Product Distribution %):")
print(product_counts)
```

**Results:**

| Product | Count | Marginal Probability P(Product) |
|:--------|------:|:-------------------------------|
| KP281 | 80 | 44.4% |
| KP481 | 60 | 33.3% |
| KP781 | 40 | 22.2% |

**Interpretation:** If a customer walks into an Aerofit store, there is approximately a 44% chance they will buy the entry-level KP281, 33% for KP481, and 22% for the advanced KP781.

### 6.6.2 Conditional Probabilities

#### Conditional Probability: P(Product | Gender)

**Question:** Given a customer's gender, what is the probability of them purchasing each product?

```python
# Conditional probability: P(Product | Gender)
cond_prob_gender = pd.crosstab(df['Gender'], df['Product'], normalize='index') * 100
print(cond_prob_gender)
```

**Results:**

| Gender | P(KP281 | Gender) | P(KP481 | Gender) | P(KP781 | Gender) |
|:-------|-------------------:|-------------------:|-------------------:|
| Female | 53.1% | 31.3% | 15.6% |
| Male | 38.8% | 34.7% | 26.5% |

**Key Finding:** Males are **3x more likely** than females to buy the high-end KP781 (26.5% vs 15.6%). Females predominantly prefer the entry-level KP281 (53.1% vs 38.8% for males).

#### Conditional Probability: P(Product | Marital Status)

```python
cond_by_marital = pd.crosstab(df['MaritalStatus'], df['Product'], normalize='index') * 100
print(cond_by_marital.round(2))
```

**Results:**

| Marital Status | P(KP281) | P(KP481) | P(KP781) |
|:---------------|:--------:|:--------:|:--------:|
| Single | 47.6% | 30.5% | 21.9% |
| Partnered | 41.9% | 35.5% | 22.6% |

**Finding:** Marital status has minimal effect on product choice. Partnered and single customers show similar purchase patterns, with only slight differences in the KP281/KP481 split.

#### Joint Probability Table

```python
# Joint probability
joint_prob = pd.crosstab(df['Gender'], df['Product'], normalize='all') * 100
print("Joint Probability Table (%):")
print(joint_prob.round(2))
```

**Example Joint Probabilities:**
- P(Male AND KP781) = ~17%
- P(Female AND KP281) = ~19%
- P(Male AND KP281) = ~26%

---

## 6.7 Correlation Analysis

```python
# Select only numeric columns
num_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_matrix = num_df.corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
```

### Correlation Results

| Feature Pair | Correlation | Strength |
|:-------------|:-----------:|:---------|
| Miles vs Usage | 0.78 | Strong positive |
| Miles vs Fitness | 0.79 | Strong positive |
| Usage vs Fitness | 0.67 | Moderate-strong positive |
| Income vs Age | 0.52 | Moderate positive |
| Income vs Miles | 0.62 | Moderate-strong positive |
| Income vs Fitness | 0.54 | Moderate positive |
| Education vs Income | 0.38 | Moderate positive |

### Key Correlation Insights

1. **Fitness, Usage, and Miles form a strongly correlated cluster** — fit people run more miles and use the treadmill more frequently
2. **Income is positively correlated with most features** — higher income customers tend to be older, fitter, and run more
3. **Education shows moderate correlation with Income** — higher education associates with higher earnings
4. **Age correlates with Income** — older customers tend to earn more

### Pairplot

```python
sns.pairplot(df, hue='Product', palette='Set1')
plt.suptitle("Pairplot: Features by Product", y=1.02)
plt.show()
```

**Key Observations from Pairplot:**
- KP781 buyers form a distinct cluster in the high-income, high-fitness, high-miles space
- KP281 buyers cluster in the lower-income, lower-fitness region
- KP481 buyers occupy the middle ground across most dimensions
- The three products show clear separation on the Income-Miles and Fitness-Miles planes

---

## 6.8 Customer Profiling

Based on the comprehensive analysis, the following customer profiles were constructed for each treadmill model:

### KP281 (Entry-Level, $1,500) — "The Beginner"

| Characteristic | Profile |
|:---------------|:--------|
| **Age** | 18-28 years (young adults) |
| **Gender** | Balanced, slight female tendency |
| **Income** | Lower range (~$30K-$45K) |
| **Education** | 14-16 years (undergraduate/early career) |
| **Fitness Level** | 2-3 (beginner to moderate) |
| **Usage** | 2-3 times per week |
| **Miles** | 50-80 miles per week |
| **Marital Status** | More singles |
| **Profile Description** | Young beginners, casual users, budget-conscious buyers looking to start a fitness routine |

### KP481 (Mid-Level, $1,750) — "The Regular"

| Characteristic | Profile |
|:---------------|:--------|
| **Age** | 25-35 years (working adults) |
| **Gender** | Balanced with slight male lean |
| **Income** | Medium range (~$45K-$60K) |
| **Education** | 16 years (graduate level) |
| **Fitness Level** | 3-4 (moderate to fit) |
| **Usage** | 3-4 times per week |
| **Miles** | 80-120 miles per week |
| **Marital Status** | Mix of single and partnered |
| **Profile Description** | Working professionals who exercise regularly, seeking quality equipment at a reasonable price |

### KP781 (Advanced, $2,500) — "The Athlete"

| Characteristic | Profile |
|:---------------|:--------|
| **Age** | 28-40 years (established professionals) |
| **Gender** | Strong male dominance (3:1 ratio) |
| **Income** | High range (~$60K-$105K) |
| **Education** | 16-20 years (graduate/post-graduate) |
| **Fitness Level** | 4-5 (fit to very fit) |
| **Usage** | 4-7 times per week |
| **Miles** | 120-360 miles per week |
| **Marital Status** | More partnered |
| **Profile Description** | Serious fitness enthusiasts and athletes with high incomes, demanding premium features and performance |

---

## 6.9 Key Insights and Findings

### Insight 1: Clear Product Tier Segmentation

The three treadmill models serve distinctly different customer segments. The separation is driven primarily by **income, fitness level, usage intensity, and age**. This validates Aerofit's three-tier product strategy.

### Insight 2: Gender is a Strong Differentiator for Premium Products

Males are 3x more likely to purchase the advanced KP781. This gender disparity is most pronounced at the premium tier and suggests different fitness investment behaviors between genders.

### Insight 3: Fitness-Usage-Miles Triangle

Fitness level, weekly usage, and expected weekly miles are strongly intercorrelated (r > 0.67). This "fitness triangle" is the strongest predictor of product tier selection — customers who rate themselves as more fit plan to use the treadmill more and expect to cover more miles, leading them to invest in premium equipment.

### Insight 4: Income is the Key Financial Barrier

There is a clear income threshold for each product tier. KP281 buyers are predominantly in the $30K-$45K range, while KP781 buyers are in the $60K+ range. This income segmentation can inform pricing and financing strategies.

### Insight 5: Marital Status Has Limited Impact

Unlike gender and income, marital status shows minimal differentiation in product choice. Both single and partnered customers show similar purchase patterns across all three tiers.

---

## 6.10 Business Recommendations

### Recommendation 1: KP281 — Target Students and Beginners

**Strategy:** Promote through student discounts, online fitness influencer partnerships, and budget-friendly bundle offers (treadmill + fitness app subscription).

**Channels:** Social media (Instagram, TikTok), university partnerships, online fitness platforms.

**Messaging:** "Start your fitness journey without breaking the bank."

### Recommendation 2: KP481 — Position as Best Value

**Strategy:** Position KP481 as the optimal price-performance option for working adults. Highlight the quality upgrade from KP281 at a modest price increase ($250).

**Channels:** Workplace wellness programs, fitness magazine ads, comparison marketing vs competitors.

**Messaging:** "The smart choice for the committed fitness enthusiast."

### Recommendation 3: KP781 — Premium Gym and Club Partnerships

**Strategy:** Market KP781 through premium gyms, health clubs, and high-end fitness retailers. Partner with professional athletes for endorsements.

**Channels:** Premium gym displays, luxury lifestyle magazines, sports events sponsorships.

**Messaging:** "Engineered for those who demand the best."

### Recommendation 4: Implement Upgrade Programs

**Strategy:** Create trade-in and upgrade programs to help customers move from KP281 to KP481 to KP781 as their fitness journey progresses.

**Program Details:**
- Offer 20% trade-in value for existing Aerofit products
- Provide installment plans for the price difference
- Send targeted upgrade offers based on usage milestones

### Recommendation 5: Gender-Specific Marketing for KP781

**Strategy:** While maintaining inclusive marketing, create specific campaigns targeting female customers for the KP781, as this segment is currently underrepresented.

**Action:** Partner with female fitness influencers, highlight female athlete testimonials, create women's fitness challenge campaigns.

### Recommendation 6: Data-Driven Customer Matching

**Strategy:** Implement an in-store or online product recommendation tool that matches customers to the right treadmill based on their age, income, fitness level, and usage goals.

**Implementation:** Simple questionnaire (5-6 questions) that maps to the customer profiles identified in this analysis.

---

## 6.11 Conclusion

The Aerofit Descriptive Statistics and Probability case study demonstrates how statistical analysis can create actionable customer intelligence from a relatively small dataset (180 customers). Key achievements include:

1. **Comprehensive customer profiling** for three product tiers using descriptive statistics and probability analysis
2. **Clear segmentation** based on demographic and behavioral characteristics
3. **Probability-based insights** that quantify the likelihood of specific purchase decisions
4. **Actionable marketing recommendations** tailored to each customer segment

The analysis confirms that Aerofit's three-tier product strategy is well-aligned with distinct customer segments. By leveraging these insights, Aerofit can create targeted marketing campaigns that increase conversion rates and customer satisfaction. The analysis scored **86.0/100**, demonstrating solid statistical and probability analysis skills.

### GitHub Repository

**Repository:** [AeroFit-Treadmill-Customer-Profiling](https://github.com/MateenahJAHAN/AeroFit-Treadmill-Customer-Profiling)
**Notebook:** `Aerofit case study solution.ipynb`
**Dataset:** `aerofit_treadmill.csv`
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Google Colab

---
