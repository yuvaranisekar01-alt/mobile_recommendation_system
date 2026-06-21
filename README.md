Product Analytics & Recommendation System

End-to-end ML pipeline that segments products via K-Means clustering and
recommends similar products via content-based & collaborative filtering,
wrapped in an interactive Streamlit app.

Pipeline Walkthrough

StepModuleWhat it does
1. Data Collection
data/generate_data.py
Builds a synthetic product catalog (price, ratings, specs, engagement)
2. Preprocessingsrc/preprocessing.py → load_data(), preprocess()
Drops duplicates, fills nulls, encodes categoricals, scales features
3. EDAsrc/eda.py → run_eda()
Brand/country distributions, correlations, top/bottom products
4. Clusteringsrc/clustering.py → run_clustering()
K-Means (k=4), PCA visualization, segment profiling
5. Recommendationssrc/recommender.py → ProductRecommender, CollaborativeFiltering
CFCosine-similarity content recs + user-based CF
6. Appapp.py
4-page Streamlit UI tying everything together

App Pages

📊 Overview & EDA — KPIs, brand/country distributions, price-vs-rating scatter, correlation heatmap
🔵 Cluster Analysis — segment summary table, PCA 2D cluster plot, feature profiles per segment, browsable product table
🎯 Recommendations — three tabs: by product ID, by desired feature profile, and collaborative filtering by user
📈 Insights & Report — price-vs-performance trends, brand comparisons, auto-generated text insights, CSV export

Notes


✨Cluster segments (Budget, Mid-range, Premium, Ultra-premium) are auto-labeled by ranking average price per cluster — relabel logic lives in assign_cluster_names() in src/clustering.py.

✨generate_synthetic_ratings() in src/recommender.py fabricates a user-ratings table so collaborative filtering has data to work with; swap in real ratings data when available.

✨Models are cached via @st.cache_data in app.py so the app doesn't retrain on every interaction — clear Streamlit's cache if you regenerate products.csv.