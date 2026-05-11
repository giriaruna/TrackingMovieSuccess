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

## Project Overview
This project investigates the relationship between industry-standard metadata and social media engagement to forecast the commercial success of modern films. We utilize machine learning to bridge the "Hype Gap" between professional industry rankings and organic audience sentiment.

### Research Question
> To what extent can YouTube marketing metrics and film metadata be integrated to predict the commercial success of a movie?

---

## Data Engineering
Data was sourced via live API calls to **TMDB** and **YouTube**. We engineered high-signal features to improve predictive accuracy:

$$ Engagement\ Ratio = \frac{Trailer\ Likes}{Trailer\ Views} $$

Movies were classified as a **Hit** ($S=1$) based on the following logic:

$$ S = \begin{cases} 1 & \text{if Rating > 7.0} \\ 0 & \text{otherwise} \end{cases} $$

---

## Machine Learning Models
1. **Random Forest Classification:** Predicts categorical success with 67% accuracy.
2. **K-Means Clustering:** Segments the market into Viral Hits, Industry Favorites, and Baseline Releases.
3. **Linear Regression:** Establishes the relationship between industry hype and audience reach.

---

## Key Findings
- **High Consistency:** A 0.90 correlation exists between YouTube views and likes.
- **The Hype Gap:** TMDB popularity reflects marketing budget (reach), but YouTube engagement reflects film quality (value).
- **Predictive Power:** Engagement metrics are statistically superior predictors of final user satisfaction compared to raw industry scores.

---

## Future Work
- **NLP Sentiment Analysis:** Analyzing the qualitative text of YouTube comments.
- **Financial Integration:** Mapping engagement decay rates to predicted box office revenue in USD.
- **Deep Learning:** Implementing Neural Networks for long-term temporal forecasting.

---
# Authors

Developed by **Aruna Giri**, **Jane Manalu**, and **KJ Moses** for the NYU Data Bootcamp Final Project.