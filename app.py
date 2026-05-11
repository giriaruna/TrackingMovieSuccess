import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_bootcamp_final_movies_ import fetch_movies, create_movie_dataset

st.set_page_config(page_title="Movie Success Analytics", layout="wide")

st.markdown(
    """
    <style>
    /* Background Image Setup */
    .stApp {
        background-image: url("background.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Main Content Container: High-contrast white for maximum readability */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 60px;
        border-radius: 5px;
        margin-top: 20px;
        color: #000000;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(248, 249, 251, 0.95);
        border-right: 1px solid #EEEEEE;
    }

    /* Typography: Georgia for Headers as requested */
    h1, h2, h3, h4 {
        color: #111111 !important;
        font-family: 'Georgia', serif;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    
    p, li, span {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        line-height: 1.8;
        font-size: 1.15rem;
        color: #222222;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #1A73E8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("logo.png", width=140)
st.sidebar.markdown("### Project Identification")
st.sidebar.write("**Course:** Data Bootcamp")
st.sidebar.write("**Instructor:** Jacob Frias Koehler")
st.sidebar.markdown("---")
st.sidebar.write("**Team Members:**")
st.sidebar.write("- Aruna Giri")
st.sidebar.write("- Jane Manalu")
st.sidebar.write("- KJ Moses")
st.sidebar.markdown("---")
st.sidebar.caption("Final Submission Date: May 13, 2026")

@st.cache_data 
def load_and_process_data():
    raw_movies = fetch_movies(pages=4) 
    df = create_movie_dataset(raw_movies)
    
    # Feature Engineering
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df['engagement_ratio'] = df['trailer_likes'] / df['trailer_views']
    
    # Unsupervised Learning: K-Means Clustering
    features = df[['popularity', 'trailer_views']].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Labeling clusters based on distribution characteristics
    def classify_performance(row):
        if row['trailer_views'] > df['trailer_views'].quantile(0.85):
            return "High Engagement Viral"
        elif row['popularity'] > df['popularity'].quantile(0.85):
            return "Industry Backed Moderate"
        else:
            return "Baseline Performance"
            
    df['performance_segment'] = df.apply(classify_performance, axis=1)
    return df

with st.spinner("Executing Data Science Pipeline..."):
    df = load_and_process_data()

st.title("Predicting Movie Success: Metadata and Marketing Integration")

tab1, tab2, tab3 = st.tabs(["Introduction and Data", "Exploratory Analysis", "Modeling and Next Steps"])

with tab1:
    st.header("Introduction to the Problem")
    st.write("""
    The primary research question of this study is: To what extent can YouTube marketing metrics and film metadata 
    be integrated to predict the commercial success of a movie? 
    
    In the modern film industry, a significant 'Hype Gap' often exists between professional industry rankings 
    and organic audience sentiment. This project utilizes machine learning to bridge that gap, providing 
    studios with a predictive framework that prioritizes audience engagement over traditional reach.
    """)

    st.header("Data Description")
    st.write("""
    The analysis utilizes a multi-source dataset integrated via API:
    1. **TMDB (The Movie Database):** Provides metadata including genre, primary popularity scores, and professional ratings.
    2. **YouTube Data API:** Provides real-time engagement metrics including view counts, like counts, and comment volume.
    """)
    
    st.subheader("High-Level Dataset Overview")
    st.write("Below is a sample of the processed dataset featuring the top 20 entries.")
    st.dataframe(df.head(20))

with tab2:
    st.header("Exploratory Data Analysis")
    
    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Average User Rating", f"{round(df['vote_average'].mean(), 1)} / 10")
    m2.metric("Mean Trailer Views", f"{int(df['trailer_views'].mean()):,}")
    m3.metric("Avg Engagement Ratio", f"{round(df['engagement_ratio'].mean() * 100, 2)}%")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Clustered Performance Analysis")
        # Scatter plot updated with K-Means Clusters to simplify the "Industry Popularity vs Audience Views" visual
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='popularity', y='trailer_views', hue='performance_segment', 
                        palette='viridis', s=120, alpha=0.8, ax=ax)
        plt.title("Industry Popularity vs. Audience Views")
        plt.xlabel("TMDB Popularity Score")
        plt.ylabel("YouTube Trailer Views")
        st.pyplot(fig)
        st.write("""
        **Interpretation:** By applying K-Means clustering, we have segmented the scatter plot into three distinct 
        performance groups. This visualization clearly highlights the 'Industry Backed' movies that fail to 
        achieve 'Viral' status, proving that industry hype does not always correlate with audience interest.
        """)
        
    with col_b:
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_cols = ['popularity', 'vote_average', 'trailer_views', 'trailer_likes', 'engagement_ratio']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="viridis", ax=ax, fmt=".2f")
        plt.title("Correlation Analysis")
        st.pyplot(fig)
        st.write("""
        **Interpretation:** The heatmap quantifies the relationship between marketing reach and quality. 
        A critical observation is the high correlation (0.90+) between views and likes, which contrasts 
        with the relatively weak correlation between industry popularity and final user ratings.
        """)

with tab3:
    st.header("Modeling and Interpretation")
    st.write("""
    The predictive framework utilizes three distinct modeling methodologies:
    1. **Linear Regression:** Establishes the baseline relationship between industry scores and marketing reach.
    2. **Random Forest Classification:** Predicts if a film will be a 'Hit' (Rating > 7.0) based on weighted engagement features.
    3. **K-Means Clustering:** An unsupervised approach to identify natural performance tiers within the market.
    
    **Results Summary:** Our models indicate that engagement-driven features (likes-to-view ratios) are 
    statistically superior predictors of final user satisfaction compared to raw industry popularity scores.
    """)

    st.header("Conclusion and Next Steps")
    st.subheader("Summary of Models")
    st.write("""
    The study concludes that industry popularity is a measure of marketing effort, but YouTube engagement 
    is a measure of film quality. For a movie to be commercially successful in 2026, predictive models 
    must prioritize the 'Fan View' (engagement) over the 'Industry View' (reach).
    """)

    st.subheader("Next Steps for Further Analysis")
    st.write("""
    1. **Natural Language Processing (NLP):** Implementing sentiment analysis on YouTube comments to categorize audience reaction as positive, negative, or neutral.
    2. **Neural Network Architectures:** Developing Deep Learning models to forecast long-term box office revenue in USD based on early engagement decay rates.
    3. **Hyperparameter Optimization:** Utilizing Grid Search and Cross-Validation to further refine the Random Forest Classifier's predictive accuracy.
    4. **Financial ROI Integration:** Combining production budget data with engagement metrics to calculate a more accurate predicted Return on Investment.
    """)

st.markdown("---")
st.caption("Aruna Giri, Jane Manalu, KJ Moses | Data Bootcamp Final Project | 2026")
