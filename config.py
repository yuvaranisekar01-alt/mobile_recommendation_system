#Paths
raw_data_path = "C:\projects\mobile_review_recommendation\data\Raw\Mobile Reviews Sentiment null.csv"
processed_dir =  "data\processed"
model_features_path = "C:\projects\mobile_review_recommendation\data\Processed\model_features.csv"
similarity_path = "C:\projects\mobile_review_recommendation\data\Processed\similarity_matrix.csv"

#segmentation
n_clusters = 4
random_state= 42

#Segment lables
segment_lables= ['Budget', 'Affordable', 'Mid-Range', 'Premium']

#Recommendation
Top_n = 5

#Features for Recommendation
recommend_features = ['price_usd', 'battery_life_rating', 'camera_rating',
    'performance_rating', 'design_rating', 'display_rating']

