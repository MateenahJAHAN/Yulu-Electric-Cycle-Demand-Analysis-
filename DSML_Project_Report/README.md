# Scaler Neovarsity | DSML Project Report

## Data Science & Machine Learning — Comprehensive Case Study Report

**Author:** Mateenah Jahan
**Program:** Scaler Neovarsity DSML (Woolf University Accredited)
**Submission Date:** February 2026
**Weighted Average Score:** 91.4/100

---

## Overview

This repository contains a comprehensive **~100-page project report** covering **5 industry-focused case studies** completed as part of the Scaler Neovarsity Data Science & Machine Learning program. Each case study demonstrates practical application of data science techniques to real-world business problems across diverse industries.

---

## Case Studies Included

| # | Case Study | Domain | Key Technique | Score |
|:-:|:-----------|:-------|:--------------|:-----:|
| 1 | **Netflix:** Data Exploration & Visualisation | Entertainment / OTT | EDA, Visualization | 95.0/100 |
| 2 | **Walmart:** Confidence Interval & CLT | Retail | CI, CLT, Statistical Inference | 94.0/100 |
| 3 | **Yulu:** Hypothesis Testing | Micro-Mobility | t-Test, ANOVA, Chi-Square | 90.0/100 |
| 4 | **Aerofit:** Descriptive Statistics & Probability | Fitness Equipment | Descriptive Stats, Probability | 86.0/100 |
| 5 | **Jamboree Education:** Linear Regression | Education / EdTech | Linear Regression, Prediction | 92.0/100 |

---

## Repository Structure

```
DSML_Project_Report/
|
|-- README.md                          # This file - Project overview & navigation
|
|-- report/                            # Complete project report
|   |
|   |-- 00_Cover_Page.md              # Cover page with student & case study details
|   |-- 01_Declaration_and_Acknowledgements.md  # Declaration, certificate, acknowledgements
|   |-- 02_Table_of_Contents.md        # Full TOC, list of figures, list of tables
|   |-- 03_Executive_Summary.md        # Executive summary of all 5 case studies
|   |-- 04_Introduction_and_Methodology.md  # Ch 1-2: Introduction + methodology overview
|   |-- 05_Comparative_Analysis_and_Conclusion.md  # Ch 8-11: Comparison + conclusion
|   |-- 06_Appendices.md               # Appendices A-D: Setup, code, formulas, glossary
|   |
|   |-- case_studies/                  # Individual case study chapters
|   |   |
|   |   |-- case_study_1_netflix/
|   |   |   |-- Netflix_Data_Exploration.md      # Ch 3: Netflix (18 pages)
|   |   |
|   |   |-- case_study_2_walmart/
|   |   |   |-- Walmart_CLT_Analysis.md          # Ch 4: Walmart (19 pages)
|   |   |
|   |   |-- case_study_3_yulu/
|   |   |   |-- Yulu_Hypothesis_Testing.md       # Ch 5: Yulu (17 pages)
|   |   |
|   |   |-- case_study_4_aerofit/
|   |   |   |-- Aerofit_Descriptive_Stats.md     # Ch 6: Aerofit (17 pages)
|   |   |
|   |   |-- case_study_5_jamboree/
|   |       |-- Jamboree_Linear_Regression.md    # Ch 7: Jamboree (13 pages)
|   |
|   |-- assets/                        # Placeholder for charts and visualizations
|       |-- .gitkeep
|
|-- notebooks/                         # Reference to original Jupyter notebooks
|   |-- .gitkeep
|
|-- data/                              # Reference to datasets used
|   |-- .gitkeep
|
|-- FULL_REPORT.md                     # Single combined file (all chapters)
```

---

## How to Navigate This Report

### For Project Reviewers

1. **Quick Overview:** Start with [Executive Summary](report/03_Executive_Summary.md) for a high-level view of all 5 case studies
2. **Individual Case Studies:** Jump to any specific case study in the `report/case_studies/` folder
3. **Full Report:** Read [FULL_REPORT.md](FULL_REPORT.md) for the complete document in one file
4. **Methodology:** See [Introduction & Methodology](report/04_Introduction_and_Methodology.md) for the analytical framework

### Reading Order (Recommended)

1. [Cover Page](report/00_Cover_Page.md)
2. [Declaration & Acknowledgements](report/01_Declaration_and_Acknowledgements.md)
3. [Table of Contents](report/02_Table_of_Contents.md)
4. [Executive Summary](report/03_Executive_Summary.md)
5. [Introduction & Methodology](report/04_Introduction_and_Methodology.md)
6. [Case Study 1: Netflix](report/case_studies/case_study_1_netflix/Netflix_Data_Exploration.md)
7. [Case Study 2: Walmart](report/case_studies/case_study_2_walmart/Walmart_CLT_Analysis.md)
8. [Case Study 3: Yulu](report/case_studies/case_study_3_yulu/Yulu_Hypothesis_Testing.md)
9. [Case Study 4: Aerofit](report/case_studies/case_study_4_aerofit/Aerofit_Descriptive_Stats.md)
10. [Case Study 5: Jamboree](report/case_studies/case_study_5_jamboree/Jamboree_Linear_Regression.md)
11. [Comparative Analysis & Conclusion](report/05_Comparative_Analysis_and_Conclusion.md)
12. [Appendices](report/06_Appendices.md)

---

## GitHub Repositories (Original Case Studies)

| Case Study | Repository Link |
|:-----------|:----------------|
| Netflix | [Netflix-Content-Strategy-Analysis](https://github.com/MateenahJAHAN/Netflix-Content-Strategy-Analysis) |
| Walmart | [Walmart-Black-Friday-Sales-Analysis](https://github.com/MateenahJAHAN/Walmart-Black-Friday-Sales-Analysis) |
| Yulu | [Yulu-Electric-Cycle-Demand-Analysis-](https://github.com/MateenahJAHAN/Yulu-Electric-Cycle-Demand-Analysis-) |
| Aerofit | [AeroFit-Treadmill-Customer-Profiling](https://github.com/MateenahJAHAN/AeroFit-Treadmill-Customer-Profiling) |
| Jamboree | Completed as part of Scaler DSML curriculum |

---

## Technical Skills Demonstrated

| Category | Skills |
|:---------|:-------|
| **Data Wrangling** | Pandas, data cleaning, type conversion, missing value treatment |
| **Visualization** | Matplotlib, Seaborn — histograms, boxplots, heatmaps, scatter plots, bar charts |
| **Statistics** | Descriptive stats, t-tests, ANOVA, chi-square, confidence intervals, CLT |
| **Probability** | Marginal, conditional, and joint probability calculations |
| **Machine Learning** | Linear regression, train-test split, model evaluation, residual analysis |
| **Tools** | Python 3.x, Google Colab, Jupyter Notebook, Git/GitHub |

---

## How to Run the Code

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### Using Google Colab (Recommended)

1. Open any case study notebook from the corresponding GitHub repository
2. Click "Open in Colab" or upload to Google Colab
3. Run all cells sequentially

### Using Local Jupyter

```bash
# Clone any case study repo
git clone https://github.com/MateenahJAHAN/Netflix-Content-Strategy-Analysis.git
cd Netflix-Content-Strategy-Analysis

# Launch Jupyter
jupyter notebook
```

---

## Contact

**Mateenah Jahan**
- Email: jahanmateenah55@gmail.com
- GitHub: [MateenahJAHAN](https://github.com/MateenahJAHAN)
- Program: Scaler Neovarsity DSML (Woolf University Accredited)

---

*This report was prepared as part of the Scaler Neovarsity Data Science & Machine Learning program requirements.*
