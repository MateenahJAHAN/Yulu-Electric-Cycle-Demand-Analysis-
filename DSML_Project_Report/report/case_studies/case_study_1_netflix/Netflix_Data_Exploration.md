# Chapter 3: Case Study 1 — Netflix: Data Exploration & Visualisation

**Score Achieved: 95.0 / 100**
**Domain: OTT Platform / Entertainment**
**Primary Technique: Exploratory Data Analysis & Data Visualization**

---

## 3.1 Problem Statement

### About Netflix

Netflix is the world's leading subscription-based streaming service, with over 230 million paid subscribers globally as of 2023. The platform offers a vast library of movies, TV shows, documentaries, and original content across multiple languages and genres. Founded in 1997 as a DVD rental service, Netflix pivoted to streaming in 2007 and has since revolutionized the entertainment industry.

### Business Problem

Netflix's executive team wants to leverage data analytics to make strategic decisions about content production and global expansion. Specifically, they need answers to two critical questions:

1. **What types of content (movies vs TV shows) should Netflix focus on producing next?**
2. **How can Netflix identify key markets and strategies for business growth in different countries?**

### Objective

Perform a comprehensive exploratory data analysis on the Netflix content catalog to uncover patterns, trends, and insights that can inform content strategy and international expansion decisions.

### Expected Outcomes

- Understand the composition of Netflix's content library (movies vs TV shows)
- Identify the top content-producing countries and their content characteristics
- Analyze temporal trends in content additions
- Profile the audience based on content ratings
- Generate actionable recommendations for content strategy and global growth

---

## 3.2 Dataset Description

### Source and Overview

The dataset contains information about all TV shows and movies available on the Netflix platform as of mid-2021. It provides a comprehensive view of the content catalog including metadata about each title.

**Dataset Dimensions:** 8,807 rows x 12 columns

### Schema

| Column | Data Type | Description | Example |
|:-------|:----------|:------------|:--------|
| `show_id` | String | Unique identifier for each title | s1, s2, s3 |
| `type` | String | Content type | Movie, TV Show |
| `title` | String | Name of the content | "Jailbirds New Orleans" |
| `director` | String | Director name(s) | "Kirsten Johnson" |
| `cast` | String | Actors featured in the content | "Brendan Fraser, Jenna Elfman" |
| `country` | String | Country of production | "United States" |
| `date_added` | String | Date the content was added to Netflix | "September 25, 2021" |
| `release_year` | Integer | Year the content was originally released | 2021 |
| `rating` | String | Content maturity rating | TV-MA, PG-13, R |
| `duration` | String | Duration (minutes for movies, seasons for TV) | "90 min", "2 Seasons" |
| `listed_in` | String | Genre(s) the content belongs to | "Dramas, International Movies" |
| `description` | String | Brief synopsis of the content | "In this documentary..." |

---

## 3.3 Data Loading and Inspection

### Initial Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(filename, encoding='latin-1')

# Display first 5 rows
display(df.head())
```

### Structural Inspection

```python
# Dataset dimensions
print(df.shape)
# Output: (8807, 12)

# Data types and non-null counts
df.info()

# Summary statistics for numerical columns
df.describe()

# Summary for categorical/text columns
df.describe(include='object')
```

### Key Observations from Initial Inspection

1. **Dataset Size:** 8,807 titles with 12 attributes
2. **Data Types:** Mostly categorical/text data with one numeric column (`release_year`)
3. **Release Year Range:** Content spans from 1925 to 2021, with an average release year around 2014
4. **Content Variety:** Multiple directors, cast members, and countries represent a diverse catalog

---

## 3.4 Data Cleaning and Missing Value Treatment

### Missing Value Analysis

```python
# Count missing values per column
df.isnull().sum()
```

**Missing Value Summary:**

| Column | Missing Count | Missing % | Treatment Strategy |
|:-------|-------------:|-----------:|:-------------------|
| `show_id` | 0 | 0.0% | No treatment needed |
| `type` | 0 | 0.0% | No treatment needed |
| `title` | 0 | 0.0% | No treatment needed |
| `director` | 2,634 | 29.9% | Retained as "Unknown" for analysis |
| `cast` | 825 | 9.4% | Retained as "Unknown" for analysis |
| `country` | 831 | 9.4% | Retained as "Unknown" for analysis |
| `date_added` | 10 | 0.1% | Rows dropped (negligible) |
| `release_year` | 0 | 0.0% | No treatment needed |
| `rating` | 4 | 0.05% | Imputed with mode |
| `duration` | 3 | 0.03% | Rows dropped (negligible) |
| `listed_in` | 0 | 0.0% | No treatment needed |
| `description` | 0 | 0.0% | No treatment needed |

### Missing Value Treatment Rationale

- **Director (29.9% missing):** High missing percentage suggests many titles lack director attribution (possibly compilations, reality shows, or incomplete metadata). Rather than dropping these rows, they were retained and flagged for analysis.
- **Cast (9.4% missing):** Similar to director, some titles may not have traditional cast information.
- **Country (9.4% missing):** Missing country data limits geographic analysis but the majority of data remains usable.
- **Date Added (0.1% missing):** Negligible — rows dropped without significant data loss.

### Data Type Conversions

```python
# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())

# Extract year and month added
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Process duration for movies (extract numeric minutes)
movies = df[df['type'] == 'Movie'].copy()
movies['duration_minutes'] = movies['duration'].str.replace(' min', '').astype(int)

# Process duration for TV shows (extract number of seasons)
tv_shows = df[df['type'] == 'TV Show'].copy()
tv_shows['num_seasons'] = tv_shows['duration'].str.replace(' Season(s)?', '', regex=True).astype(int)
```

---

## 3.5 Exploratory Data Analysis

### 3.5.1 Univariate Analysis

#### Distribution of Content Type (Movies vs TV Shows)

```python
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df, palette='pastel')
plt.title("Distribution of Content Type (Movies vs TV Shows)")
plt.show()
```

**Findings:**
- **Movies:** ~6,131 titles (69.6% of total catalog)
- **TV Shows:** ~2,676 titles (30.4% of total catalog)
- Movies clearly dominate Netflix's content library, outnumbering TV shows by more than 2:1
- However, this ratio has been shifting — TV show additions have accelerated significantly since 2015

#### Distribution of Content Ratings

```python
plt.figure(figsize=(10, 5))
sns.countplot(y='rating', data=df, order=df['rating'].value_counts().index, palette='Set2')
plt.title("Distribution of Content Ratings")
plt.show()
```

**Findings:**
- **TV-MA (Mature Audience)** is the most common rating, indicating Netflix heavily targets adult viewers
- **TV-14** is the second most common, catering to teenage audiences
- **TV-PG** and **PG-13** follow, serving general/family audiences
- Content rated **TV-Y** (young children) and **TV-G** (general audience) are significantly underrepresented
- This reveals a strategic gap in children's and family content

#### Distribution of Release Years

```python
plt.figure(figsize=(12, 5))
sns.histplot(df['release_year'], bins=30, kde=False, color='skyblue')
plt.title("Distribution of Release Years")
plt.show()
```

**Findings:**
- The vast majority of Netflix content was released after 2000
- There is a dramatic spike in content from 2015-2020
- Peak content production occurred around 2017-2018
- Historical content (pre-2000) forms a very small fraction of the catalog
- This indicates Netflix prioritizes modern, relevant content

#### Distribution of Release Years (KDE)

```python
plt.figure(figsize=(10, 5))
sns.kdeplot(df['release_year'], shade=True, color='purple')
plt.title("Distribution of Release Years")
plt.xlabel("Year")
plt.show()
```

The KDE plot confirms the right-heavy distribution with the density peak clearly centered on recent years (2015-2020).

#### Top 10 Countries by Content Count

```python
plt.figure(figsize=(10, 5))
df['country'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Countries by Content Count")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.show()
```

**Findings:**
- **United States** dominates with the highest content count by a significant margin
- **India** is the second-largest content producer, reflecting Netflix's investment in the Indian market
- **United Kingdom, Japan, and South Korea** round out the top 5
- Notable presence from **Canada, Spain, France, Turkey, and Australia**
- This distribution highlights Netflix's focus on English-language and Indian content

---

### 3.5.2 Bivariate Analysis

#### Content Added Over Time (Movies vs TV Shows)

```python
# Analyze content additions by year
yearly_content = df.groupby(['year_added', 'type']).size().reset_index(name='count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_content, x='year_added', y='count', hue='type', marker='o')
plt.title("Content Added to Netflix Over Time")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.show()
```

**Findings:**
- Both movies and TV shows show exponential growth in additions from 2015 onwards
- Movie additions peaked around 2019 before experiencing a slight decline (possibly due to COVID-19)
- TV show additions have been growing more rapidly as a percentage, indicating Netflix's strategic shift toward series content
- The gap between movie and TV show additions has been narrowing over time

#### Rating Distribution by Content Type

```python
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='rating', hue='type',
              order=df['rating'].value_counts().index,
              palette='Set2')
plt.title("Rating Distribution by Content Type")
plt.show()
```

**Findings:**
- TV-MA content is predominantly movies
- TV-14 shows a more balanced split between movies and TV shows
- TV-PG content has a higher proportion of TV shows
- R-rated content is almost exclusively movies
- This suggests different content strategies for different maturity levels

#### Content Type Distribution by Top Countries

```python
# Analyze content type by country
top_countries = df['country'].value_counts().head(10).index
country_type = df[df['country'].isin(top_countries)]
ct_table = pd.crosstab(country_type['country'], country_type['type'])
ct_table.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title("Content Type Distribution by Top 10 Countries")
plt.ylabel("Number of Titles")
plt.show()
```

**Findings:**
- The US produces a mix of both movies and TV shows, but movies dominate
- India produces primarily movies with very few TV shows
- Japan and South Korea produce a higher proportion of TV shows (anime and K-dramas)
- The UK produces a mix similar to the US pattern
- Each country has a unique content production profile that Netflix can leverage

---

### 3.5.3 Multivariate Analysis

#### Genre Analysis

```python
# Explode genres for analysis
df_genres = df.copy()
df_genres['genre'] = df_genres['listed_in'].str.split(', ')
df_genres = df_genres.explode('genre')

# Top 15 genres
plt.figure(figsize=(12, 6))
df_genres['genre'].value_counts().head(15).plot(kind='barh', color='teal')
plt.title("Top 15 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.show()
```

**Findings:**
- **International Movies** is the top genre, reflecting Netflix's global content strategy
- **Dramas** and **Comedies** are the next most popular genres
- **International TV Shows** ranks high, confirming the global expansion strategy
- Documentary and stand-up comedy content form a significant portion
- Niche genres like anime, horror, and reality TV have dedicated but smaller catalogs

#### Content Added by Month

```python
# Monthly addition pattern
plt.figure(figsize=(12, 6))
df['month_added'].value_counts().sort_index().plot(kind='bar', color='coral')
plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Number of Titles")
plt.show()
```

**Findings:**
- Content additions peak in **December and January**, likely tied to holiday season viewing
- A secondary peak occurs in **July**, coinciding with summer breaks
- The lowest addition periods are **February and May**
- This suggests a strategic content release calendar aligned with viewer availability

#### Heatmap: Content Added by Year and Month

```python
pivot = df.pivot_table(index='month_added', columns='year_added',
                       values='show_id', aggfunc='count')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title("Content Added to Netflix (Month x Year)")
plt.show()
```

This heatmap provides a comprehensive view of Netflix's content addition strategy across time, clearly showing the acceleration in content additions from 2016 onwards with seasonal patterns.

---

## 3.6 Key Insights and Findings

### Insight 1: Movies Dominate, But TV Shows Are Gaining Ground

Netflix's catalog is approximately 70% movies and 30% TV shows. However, the rate of TV show additions has been accelerating since 2015, indicating a strategic pivot. This aligns with industry trends where serialized content drives higher engagement and user retention through binge-watching behavior.

**Supporting Evidence:**
- Movie count: ~6,131 titles (69.6%)
- TV Show count: ~2,676 titles (30.4%)
- TV show addition growth rate exceeds movie addition growth rate since 2016

### Insight 2: The United States and India Are Dominant Content Hubs

The US produces the most Netflix content by a significant margin, followed by India. This reflects both Netflix's American origins and its aggressive expansion in the Indian market (one of its largest subscriber bases globally).

**Supporting Evidence:**
- US content constitutes the largest single-country contribution
- India is second, primarily contributing movies
- Combined, these two countries represent over 50% of attributable content

### Insight 3: Target Audience Skews Towards Adults and Teenagers

The predominance of TV-MA and TV-14 ratings indicates that Netflix's content strategy primarily targets mature audiences. There is a notable underrepresentation of content for young children and families.

**Supporting Evidence:**
- TV-MA is the most common rating across both movies and TV shows
- TV-Y and TV-G rated content together make up less than 5% of the catalog
- This represents both a strategic focus and a potential growth opportunity

### Insight 4: Content Strategy Favors Recent Releases

Over 80% of Netflix content was released after 2010, with the majority concentrated in the 2015-2020 period. This indicates a clear preference for modern, contemporary content over classic or vintage titles.

**Supporting Evidence:**
- Mean release year: ~2014
- Median release year: ~2017
- Content from before 2000 makes up a very small fraction

### Insight 5: Seasonal Content Release Patterns Exist

Netflix demonstrates strategic content release timing, with peak additions during holiday seasons (December-January) and summer months (July). This aligns with periods of maximum viewer availability and engagement.

**Supporting Evidence:**
- Highest content additions: December, January, July
- Lowest content additions: February, May
- Pattern consistent across multiple years

### Insight 6: Genre Diversity Reflects Global Strategy

The prominence of "International Movies" and "International TV Shows" as top genres confirms Netflix's commitment to building a globally diverse content library. This supports localized content strategies across different markets.

**Supporting Evidence:**
- "International Movies" is the single largest genre category
- Dramas and Comedies are universally popular across markets
- Regional genres (anime from Japan, K-dramas from Korea) have dedicated catalogs

---

## 3.7 Business Recommendations

Based on the comprehensive analysis of Netflix's content catalog, the following actionable recommendations are proposed:

### Recommendation 1: Accelerate TV Show Investment

**Rationale:** TV shows drive higher engagement metrics through binge-watching behavior and create longer-term subscriber retention. The growing share of TV show additions should be further accelerated.

**Action Items:**
- Increase original series commissioning by 20-30% annually
- Focus on short-season formats (6-10 episodes) that reduce production costs while maintaining engagement
- Prioritize genres that perform well in series format: crime, thriller, drama, and reality

### Recommendation 2: Expand Kids and Family Content

**Rationale:** The significant underrepresentation of TV-Y, TV-G, and PG-rated content represents a substantial growth opportunity. Family subscriptions typically have lower churn rates and higher lifetime value.

**Action Items:**
- Double the investment in children's original content over the next 2 years
- Acquire licensing rights for popular animated franchises
- Develop educational content partnerships to differentiate from competitors
- Create a stronger kids' content brand identity within the platform

### Recommendation 3: Diversify International Content Sources

**Rationale:** While the US and India dominate content production, there are underserved markets with high subscriber growth potential, particularly in Europe, Latin America, and Africa.

**Action Items:**
- Establish content production partnerships in Brazil, Germany, France, and Nigeria
- Increase subtitling and dubbing capabilities to make international content accessible
- Invest in local production studios in high-growth markets
- Leverage the success of K-dramas and anime as models for other regional content

### Recommendation 4: Improve Metadata Quality

**Rationale:** With nearly 30% of titles missing director information and ~9% missing country data, the recommendation engine's effectiveness is significantly impaired.

**Action Items:**
- Implement a metadata enrichment program to fill missing director, cast, and country data
- Partner with external databases (IMDb, TMDb) for automated metadata completion
- Establish stricter metadata requirements for new content additions
- This improvement will directly enhance search functionality and personalized recommendations

### Recommendation 5: Optimize Content Release Strategy

**Rationale:** The data shows clear seasonal patterns in content additions. Aligning high-quality content releases with peak viewing periods can maximize engagement.

**Action Items:**
- Schedule tentpole releases (major original movies/series) during December-January and July
- Maintain a steady cadence of content additions during traditionally slower periods (February, May)
- Use data-driven A/B testing to optimize release timing for different genres

### Recommendation 6: Double Down on Popular Genres

**Rationale:** Dramas, Comedies, and International content are consistently popular across markets and should remain core investment areas.

**Action Items:**
- Maintain and increase investment in drama and comedy productions
- Develop cross-genre content that combines popular categories (e.g., comedy-drama, thriller-drama)
- Create genre-specific content hubs within the platform for improved discoverability

---

## 3.8 Conclusion

The Netflix Data Exploration and Visualisation case study demonstrates the power of systematic exploratory data analysis in uncovering actionable business insights from a large content catalog. Through careful data cleaning, comprehensive univariate and bivariate analysis, and thoughtful interpretation, several strategic opportunities were identified:

1. The platform's content strategy is evolving from movie-dominated to a more balanced movie-TV show mix
2. Geographic content distribution reveals both strengths (US, India) and expansion opportunities
3. Audience targeting skews adult/teen, with family content representing an untapped segment
4. Content release timing follows strategic seasonal patterns that can be further optimized

These findings provide a data-driven foundation for Netflix's content investment decisions, international expansion strategy, and platform optimization efforts. The analysis scored **95.0/100**, reflecting strong analytical execution and insightful business recommendations.

### GitHub Repository

**Repository:** [Netflix-Content-Strategy-Analysis](https://github.com/MateenahJAHAN/Netflix-Content-Strategy-Analysis)
**Notebook:** `Netflix case study.ipynb`
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Google Colab

---
