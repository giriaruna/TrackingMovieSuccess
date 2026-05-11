# Predicting Movie Success: An Integration of TMDB Metadata and YouTube Engagement Metrics

## Team Members
- Aruna Giri  
- Jane Manalu  
- KJ Moses  

**Course:** Data Bootcamp Final  
**Instructor:** Jacob Frias Koehler  
**Date:** May 13, 2026  
**Live Application:** [tracking-movie-success.streamlit.app](https://tracking-movie-success.streamlit.app/)

---

# Introduction

The modern film industry faces a significant challenge in predicting commercial success prior to theatrical release. Traditional industry metrics often focus heavily on reach and marketing budget, often creating a **"Hype Gap"** between professional expectations and authentic audience sentiment.

This project explores the following research question:

> **To what extent can YouTube marketing metrics and film metadata be integrated to predict the commercial success of a movie?**

---

# Data Description

The dataset was constructed using live API integration from two primary sources:

## Data Sources

### The Movie Database (TMDB)
Provided metadata including:
- Genre
- Popularity scores
- Professional ratings
- Release information

### YouTube Data API
Provided real-time audience engagement metrics including:
- Trailer view counts
- Like counts
- Comment volume

---

# Feature Engineering

The primary engineered feature used in this study is the **Engagement Ratio**:

$$Engagement\ Ratio = \frac{Trailer\ Likes}{Trailer\ Views}$$

This metric was designed to measure audience interaction quality rather than raw exposure.

---

# Models and Methods

We implemented a multi-model machine learning pipeline to analyze the dataset from several mathematical perspectives.

---

## 1. Linear Regression

To identify baseline trends between industry popularity ($x$) and audience views ($y$), we utilized the linear model:

$$
y=\beta_0+\beta_1x+\epsilon
$$

Where:
- $\beta_0$ = intercept
- $\beta_1$ = weight of industry hype on audience reach
- $\epsilon$ = random error term

The goal was to evaluate whether professional popularity metrics translate into measurable audience engagement.

---

## 2. Random Forest Classification

A **Random Forest Classifier** was trained to predict a binary **"Hit"** status.

A movie is classified as successful if:

$$
S=
\begin{cases}
1 & \text{if Rating > 7.0} \\
0 & \text{otherwise}
\end{cases}
$$

The classifier leveraged:
- TMDB popularity
- Trailer views
- Likes
- Engagement ratio
- Comment activity

This allowed us to compare traditional industry signals against audience-driven engagement metrics.

---

## 3. K-Means Clustering

We applied **unsupervised learning** using K-Means clustering to segment films into natural performance groups, including:

- Viral Hits
- Industry Favorites
- Standard Releases

Scaled numerical features were used to identify hidden behavioral patterns within movie performance data.

---

# Results and Interpretation

Our findings indicate that **audience engagement metrics are statistically superior predictors of final user satisfaction compared to raw industry popularity scores.**

Key observations include:

- Strong correlation (**0.90**) between trailer views and likes
- Weak relationship between TMDB popularity and final audience ratings
- Engagement ratios better reflected authentic audience sentiment
- Industry popularity primarily measured marketing reach rather than perceived quality

These findings support the hypothesis that organic interaction metrics provide stronger predictive value than traditional promotional indicators.

---

# Conclusion

This project demonstrates an important distinction in modern entertainment analytics:

- **Industry popularity** measures marketing effort
- **YouTube engagement** measures audience value

In the evolving 2026 media landscape, predictive movie models should prioritize audience engagement metrics over simple exposure-based indicators.

---

# Future Work

## Sentiment Analysis
Implement Natural Language Processing (NLP) techniques to analyze YouTube comments and extract qualitative audience sentiment.

## Financial Forecasting
Develop Neural Network models capable of forecasting box office revenue using early engagement decay patterns.

## Budget Integration
Incorporate production and marketing budgets to calculate ROI-driven success metrics.

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- TMDB API
- YouTube Data API

---

# Authors

Developed by **Aruna Giri**, **Jane Manalu**, and **KJ Moses** for the NYU Data Bootcamp Final Project.