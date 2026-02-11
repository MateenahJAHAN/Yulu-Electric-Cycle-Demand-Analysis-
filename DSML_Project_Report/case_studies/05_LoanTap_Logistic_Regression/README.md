# Case Study 5: LoanTap - Logistic Regression

**Score:** 69.0/100 | **Status:** Completed | **Due Date:** 12 Feb 2026

## Overview
Predictive analytics framework for loan default prediction using logistic regression
on asymmetric/imbalanced data from LoanTap.

## Business Problem
Predict loan default (Charged Off) using borrower and credit attributes to reduce
Non-Performing Assets (NPAs) without rejecting credit-worthy applicants.

## Dataset
- Shape: 3,000 rows x 27 columns
- Target: loan_status (Fully Paid: 91.8%, Charged Off: 8.2%)
- Key features: loan_amnt, term, int_rate, installment, grade, emp_title,
  home_ownership, annual_inc, dti, revol_util, pub_rec

## Techniques Used
- Logistic Regression with class weighting
- Feature Engineering (binary risk flags)
- Data Preprocessing (scaling, encoding, outlier capping)
- ROC AUC and Precision-Recall evaluation
- Threshold optimization

## Tools
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Key Results
- ROC AUC: 0.574
- Accuracy: 78.3%
- Default Recall: 24.5%
- Key risk factors: int_rate, dti, revol_util, 60-month term

## GitHub Repository
https://github.com/MateenahJAHAN/Predictive-Analytics-in-Asymmetric-Data-A-Robust-Classification-Framework
