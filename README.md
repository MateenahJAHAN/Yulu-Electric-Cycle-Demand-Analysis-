#  Yulu Electric Cycle Demand Analysis  

 

---

##  Project Overview  

This project analyzes the **factors influencing the demand for shared electric cycles** in India using the **Yulu dataset**.  
Yulu is India’s leading micro-mobility provider offering shared electric bikes to solve first and last-mile connectivity.  

Recently, Yulu experienced a drop in revenue.  
The goal of this analysis is to **identify which variables significantly affect demand** and **how strongly they describe the rental patterns**.  

---

##  Objectives  

1. Identify variables (like season, weather, working days) that influence Yulu cycle demand.  
2. Test if the number of cycles rented differs by season, weather, or working day.  
3. Check if predictor variables like **Weather** and **Season** are related.  
4. Provide insights to help Yulu optimize operations and increase revenue.  

---

##  Dataset Information  

| Column | Description |
|--------|-------------|
| datetime | Date and time of record |
| season | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| holiday | 1: Holiday, 0: Non-Holiday |
| workingday | 1: Working day, 0: Weekend/Holiday |
| weather | Weather condition (1 = Clear → 4 = Heavy Rain/Snow) |
| temp | Temperature in Celsius |
| atemp | “Feels like” temperature |
| humidity | Humidity level |
| windspeed | Wind speed |
| casual | Count of casual users |
| registered | Count of registered users |
| count | Total count of rented bikes |

📊 Dataset size: **10,886 rows × 12 columns**  
✅ No missing values were found.

---

## 🔍 Steps Performed  

### 1️⃣ Problem Understanding  
Defined the business goal — find **which variables affect Yulu demand** and whether there’s a relationship between weather, working days, and seasons.

---

### 2 Exploratory Data Analysis (EDA)  
- Checked dataset structure, data types, and missing values.  
- Converted categorical variables (`season`, `holiday`, `workingday`, `weather`) into categorical type.  
- Generated summary statistics for numerical features.  
- Created visualizations:  
  - **Univariate:** Distribution of `temp`, `humidity`, `windspeed`, `count`  
  - **Bivariate:** Relationship between `count` vs `season`, `weather`, `workingday`

---

### 3 Hypothesis Testing  

We performed **three statistical tests**:  

####  (a) Two-Sample t-Test  
**Goal:** Check if cycle rentals differ between working days and non-working days.  
- **H₀:** No difference in means  
- **H₁:** Means are different  
- **Result:** p = 0.216 > 0.05 →  Fail to reject H₀  
**Conclusion:** Working day has no significant effect on rentals.

---

####  (b) ANOVA Test  

**Goal 1:** Rentals across different **Seasons**  
- F = 236.95, p ≈ 6.16e-149 →  Significant difference  

**Goal 2:** Rentals across different **Weather conditions**  
- F = 98.28, p ≈ 4.97e-43 →  Significant difference  

**Conclusion:**  
Rental demand **varies significantly** with both **Season** and **Weather**.

---

####  (c) Chi-Square Test  

**Goal:** Check if **Weather** depends on **Season**  
- χ² = 49.16, df = 9, p = 1.55e−07 →  Significant  
**Conclusion:** Weather and Season are **related**.

---

##  Key Insights  

1. **Weather** and **Season** strongly affect Yulu bike demand.  
2. **Working day** does **not** significantly affect rentals.  
3. **Highest rentals** occur in clear or pleasant weather.  
4. **Lowest rentals** occur in rain or extreme weather.  
5. **Weather and Season** are statistically related.  

---

## 📊 Statistical Summary  

| Test | Variables | p-value | Significance | Conclusion |
|------|------------|----------|---------------|-------------|
| Two-sample t-test | Working Day vs Count | 0.216 |  Not Significant | Working day doesn’t affect demand |
| ANOVA | Season vs Count | 6.16e-149 |  Significant | Rentals vary by season |
| ANOVA | Weather vs Count | 4.97e-43 |  Significant | Rentals vary by weather |
| Chi-square | Weather vs Season | 1.55e-07 | Significant | Weather depends on season |

---

##  Tools and Libraries Used  

- **Python**  
- **Pandas**, **NumPy** – data cleaning & analysis  
- **Matplotlib**, **Seaborn** – visualization  
- **Scipy.stats** – statistical hypothesis testing  

---

##  Conclusion  

- Yulu demand is **significantly impacted** by **season** and **weather**.  
- Working days have **no major effect**.  
- These insights help Yulu optimize resource allocation & marketing based on conditions.

---

##  Future Work  

- Apply **time series forecasting** for demand prediction.  
- Incorporate **pricing strategy analysis**.  
- Include **demographic and behavioral variables** for deeper insights.  

---

##  Project Structure  
Yulu_CaseStudy/

│
├── yulu_data.csv # Dataset

├── Yulu.ipynb # Full analysis notebook

├── README.md # Project documentation

└── output_graphs/ # Visualizations

------yulu pdf # Google collab


---

###  Author  

**Mateenah Jahan**  
 [jahanmateenah55@gmail.com](mailto:jahanmateenah55@gmail.com)  
 Researcher | Passionate about Data Science and Machine Learning  

---



