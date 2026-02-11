# Case Study 3: Yulu - Hypothesis Testing

**Score:** 90.0/100 | **Status:** Completed | **Due Date:** 11 Nov 2025

## Overview
Statistical analysis of factors influencing the demand for Yulu shared electric cycles
in India using hypothesis testing methods.

## Business Problem
Yulu experienced a drop in revenue and needs to identify which variables significantly
affect demand and how strongly they describe rental patterns.

## Dataset
- Shape: 10,886 rows x 12 columns
- Key columns: datetime, season, holiday, workingday, weather, temp, atemp,
  humidity, windspeed, casual, registered, count
- No missing values

## Techniques Used
- Two-Sample t-Test
- One-Way ANOVA
- Chi-Square Test of Independence
- Exploratory Data Analysis

## Tools
- Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy.stats

## Key Findings
- Weather and Season strongly affect bike demand (ANOVA p < 0.001)
- Working day has NO significant effect on rentals (t-test p = 0.216)
- Weather and Season are statistically related (Chi-Square p < 0.001)
- Highest rentals occur in clear/pleasant weather

## GitHub Repository
https://github.com/MateenahJAHAN/Yulu-Electric-Cycle-Demand-Analysis-
