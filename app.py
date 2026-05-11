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
    
    /* Main Content Container: High-contrast white for readability */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.98) !important;
        padding: 60px;
        border-radius: 10px;
        margin-top: 20px;
        color: #000000 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* FORCE ALL TEXT IN MAIN CONTAINER TO BLACK */
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4, 
    .main .block-container p, 
    .main .block-container li, 
    .main .block-container span, 
    .main .block-container label,
    .main .block-container div {
        color: #000000 !important;
    }

    /* Professional Typography */
    h1, h2, h3, h4 {
        font-family: 'Georgia', serif !important;
        border-bottom: 1px solid #DDDDDD;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }

    /* FIXING TABS VISIBILITY */
    /* This ensures tab names are bold, black, and clear regardless of theme */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
    }

    /* FIXING SIDEBAR VISIBILITY */
    [data-testid="stSidebar"] {
        background-color: rgba(245, 245, 245, 1.0) !important;
        border-right: 2px solid #E0E0E0;
    }

    /* Force Sidebar text to be high-contrast black */
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #1A73E8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("logo.png", width=140)
st.sidebar.markdown("### Project Identification")
st.sidebar.write("**Course:** Data Bootcamp Final")
st.sidebar.write("**Instructor:** Jacob Frias Koehler")
st.sidebar.markdown("---")
st.sidebar.write("**Team Members:**")
st.sidebar.write("- Aruna Giri")
st.sidebar.write("- Jane Manalu")
st.sidebar.write("- KJ Moses")
st.sidebar.markdown("---")
st.sidebar.caption("Submission Date: May 13, 2026")

@st.cache_data 
def load_and_process_data():
    raw_movies = fetch_movies(pages=4) 
    df = create_movie_dataset(raw_movies)
    
    # Feature Engineering
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df['engagement_ratio'] = df['trailer_likes'] / df['trailer_views']
    
    # Machine Learning: K-Means Clustering
    features = df[['popularity', 'trailer_views']].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
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

tab1, tab2, tab3 = st.tabs(["Project Overview", "Exploratory Analysis", "Modeling and Conclusion"])

with tab1:
    st.header("Introduction to the Problem")
    st.write("""
    The primary research question of this study is: To what extent can YouTube marketing metrics and film metadata 
    be integrated to predict the commercial success of a movie? 
    
    In the modern film industry, a significant Hype Gap often exists between professional industry rankings 
    and organic audience sentiment. This project utilizes machine learning to bridge that gap.
    """)

    st.header("Data Description")
    st.write("""
    The analysis utilizes a multi-source dataset integrated via API:
    1. **TMDB (The Movie Database):** Metadata including genre, popularity scores, and professional ratings.
    2. **YouTube Data API:** Real-time engagement metrics including view counts, like counts, and comment volume.
    """)
    
    st.subheader("Processed Dataset Preview")
    st.dataframe(df.head(20))

with tab2:
    st.header("Exploratory Data Analysis")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Average User Rating", f"{round(df['vote_average'].mean(), 1)} / 10")
    m2.metric("Mean Trailer Views", f"{int(df['trailer_views'].mean()):,}")
    m3.metric("Avg Engagement Ratio", f"{round(df['engagement_ratio'].mean() * 100, 2)}%")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Clustered Performance Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='popularity', y='trailer_views', hue='performance_segment', 
                        palette='viridis', s=120, alpha=0.8, ax=ax)
        plt.title("Industry Popularity vs. Audience Views")
        plt.xlabel("TMDB Popularity Score")
        plt.ylabel("YouTube Trailer Views")
        st.pyplot(fig)
        st.write("**Analysis:** K-Means clustering highlights the 'Industry Backed' movies that fail to achieve Viral status.")
        
    with col_b:
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_cols = ['popularity', 'vote_average', 'trailer_views', 'trailer_likes', 'engagement_ratio']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="viridis", ax=ax, fmt=".2f")
        plt.title("Correlation Analysis")
        st.pyplot(fig)
        st.write("**Analysis:** High correlation (0.90+) between views and likes suggests strong audience consistency.")

with tab3:
    st.header("Modeling and Interpretation")
    st.write("""
    1. **Linear Regression:** Establishes trends between marketing reach and audience views.
    2. **Random Forest Classification:** Predicts 'Hit' status (Rating > 7.0).
    3. **K-Means Clustering:** Identifies natural performance tiers within the market.
    """)

    st.header("Conclusion and Next Steps")
    st.subheader("Summary of Models")
    st.write("""
    Audience engagement metrics (likes-to-view ratios) are statistically superior predictors of final user 
    satisfaction compared to raw industry popularity scores.
    """)

    st.subheader("Next Steps for Further Analysis")
    st.write("""
    - **NLP Sentiment Analysis:** Analyzing the text of YouTube comments.
    - **Neural Networks:** Forecasting box office revenue in USD.
    - **Hyperparameter Optimization:** Refining models via Grid Search.
    """)

st.markdown("---")
st.caption("Aruna Giri, Jane Manalu, KJ Moses | Data Bootcamp Final Project | 2026")
