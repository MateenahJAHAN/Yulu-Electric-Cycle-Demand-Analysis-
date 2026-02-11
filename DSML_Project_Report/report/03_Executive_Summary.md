# Executive Summary

---

## Overview

This comprehensive project report presents the findings from five industry-focused case studies completed as part of the **Scaler Neovarsity Data Science & Machine Learning (DSML) Program**. Each case study addresses a real-world business problem using data-driven methodologies, encompassing the complete data science lifecycle from problem formulation to actionable business recommendations.

The five case studies span diverse industries — entertainment (Netflix), retail (Walmart), micro-mobility (Yulu), fitness equipment (Aerofit), and education (Jamboree Education) — demonstrating the versatility and power of data science techniques in solving complex business challenges.

---

## Case Study Summaries

### Case Study 1: Netflix — Data Exploration & Visualisation (Score: 95.0/100)

**Objective:** Analyze the Netflix content catalog (8,807 titles) to identify content strategy patterns and provide data-driven recommendations for global expansion.

**Key Techniques:** Exploratory Data Analysis, Univariate and Bivariate Analysis, Data Visualization, Missing Value Treatment.

**Key Finding:** Movies constitute approximately 70% of Netflix's catalog, but TV show additions have accelerated since 2015. The US and India dominate content production, with TV-MA and TV-14 being the most common ratings. A significant gap exists in children's and family content.

**Business Impact:** Recommendations to invest more in TV show production, expand kids/family content, diversify international content sources, and improve metadata quality for better recommendation engine performance.

---

### Case Study 2: Walmart — Confidence Interval & CLT (Score: 94.0/100)

**Objective:** Analyze 550,068 Black Friday transaction records to understand how customer spending varies by gender, age group, and marital status, and to estimate population-level spending patterns using statistical inference.

**Key Techniques:** Confidence Intervals (Analytic and Bootstrap), Central Limit Theorem, Hypothesis Testing, Non-Graphical and Graphical EDA.

**Key Finding:** Male customers spend approximately Rs 700 more per transaction than female customers (Rs 9,438 vs Rs 8,735), with non-overlapping 95% confidence intervals confirming statistical significance. Marital status has negligible impact on spending, while the 26-50 age group represents the highest-value customer segment.

**Business Impact:** Gender-targeted marketing strategies, premium product promotions for male customers, loyalty programs for high-value age segments, and increased inventory focus on City Category B locations.

---

### Case Study 3: Yulu — Hypothesis Testing (Score: 90.0/100)

**Objective:** Identify the significant factors affecting demand for Yulu's shared electric cycles using 10,886 hourly rental records, applying rigorous statistical hypothesis testing.

**Key Techniques:** Two-Sample t-Test, One-Way ANOVA, Chi-Square Test of Independence, Bivariate Analysis.

**Key Finding:** Season (F = 236.95, p = 6.16e-149) and Weather (F = 98.28, p = 4.97e-43) significantly affect bike rental demand, while Working Day does not (p = 0.216). Weather conditions are statistically dependent on season (Chi-square = 49.16, p = 1.55e-07).

**Business Impact:** Seasonal and weather-based resource allocation strategies, dynamic pricing models, and optimized fleet distribution to maximize revenue during favorable conditions.

---

### Case Study 4: Aerofit — Descriptive Statistics & Probability (Score: 86.0/100)

**Objective:** Profile customers of three treadmill models (KP281, KP481, KP781) using data from 180 customers to create targeted marketing strategies for each product tier.

**Key Techniques:** Descriptive Statistics, Marginal and Conditional Probability, Customer Segmentation, Outlier Detection, Correlation Analysis.

**Key Finding:** 44% of customers purchased the entry-level KP281, with males being 3x more likely to purchase the advanced KP781. Customer profiles differ significantly by age, income, fitness level, and usage patterns across treadmill tiers.

**Business Impact:** Tier-specific marketing strategies — student discounts for KP281, value positioning for KP481 working adults, and premium gym partnerships for KP781. Trade-in and upgrade programs to move customers up the product ladder.

---

### Case Study 5: Jamboree Education — Linear Regression (Score: 92.0/100)

**Objective:** Build a predictive model for graduate school admission chances using academic and profile features, enabling Jamboree Education to provide data-driven counseling to students.

**Key Techniques:** Linear Regression, Feature Engineering, Model Evaluation (R-squared, RMSE, MAE), Residual Analysis, Multicollinearity Assessment (VIF).

**Key Finding:** CGPA, GRE Score, and TOEFL Score are the strongest predictors of admission probability, with the linear regression model achieving an R-squared of approximately 0.82. Research experience provides a notable positive boost to admission chances.

**Business Impact:** Data-driven student counseling recommendations, optimized preparation strategies focusing on high-impact factors, and personalized guidance based on predicted admission probability.

---

## Technical Skills Demonstrated

This report demonstrates proficiency in the following areas:

| Skill Category | Techniques Applied |
|:---------------|:-------------------|
| **Data Wrangling** | Data loading, cleaning, type conversion, missing value treatment, outlier handling |
| **Exploratory Data Analysis** | Univariate, bivariate, and multivariate analysis; distribution analysis |
| **Statistical Inference** | t-Tests, ANOVA, Chi-Square, Confidence Intervals, CLT |
| **Probability** | Marginal, conditional, and joint probability calculations |
| **Machine Learning** | Linear Regression, feature selection, model evaluation |
| **Data Visualization** | Histograms, boxplots, heatmaps, bar charts, scatter plots, pair plots |
| **Business Analytics** | Customer segmentation, profiling, actionable recommendation formulation |

---

## Tools and Technologies

- **Programming Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistical Analysis:** Scipy.stats
- **Machine Learning:** Scikit-learn, Statsmodels
- **Development Environment:** Google Colab, Jupyter Notebook
- **Version Control:** Git, GitHub

---

## Aggregate Performance

| Case Study | Score | Domain | Primary Technique |
|:-----------|------:|:-------|:------------------|
| Netflix | 95.0/100 | Entertainment | EDA & Visualization |
| Walmart | 94.0/100 | Retail | CI & CLT |
| Jamboree Education | 92.0/100 | Education | Linear Regression |
| Yulu | 90.0/100 | Transport | Hypothesis Testing |
| Aerofit | 86.0/100 | Fitness | Descriptive Stats & Probability |
| **Weighted Average** | **91.4/100** | | |

---

The following chapters present each case study in full detail, including problem context, data exploration, analytical methodology, findings, and business recommendations.

---
