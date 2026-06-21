import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

recommend_features = ['price_usd', 'battery_life_rating', 'camera_rating',
    'performance_rating', 'design_rating', 'display_rating']

segment_order = ['Budget', 'Affordable', 'Mid-Range', 'Premium']

def build_similarity_matrix(df_model):
    print("\n[Recommender] Building similarity matrix...")
    

    #scaling features 
    X= df_model[recommend_features].values
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(X)
    feature_scaled[:, 0] = feature_scaled[:, 0] * 3  #price weight

    #similarity matrix
    sim_matrix = cosine_similarity(feature_scaled)
    sim_df = pd.DataFrame(sim_matrix, index=df_model['model'].str.strip().values,
                       columns=df_model['model'].str.strip().values )
    
    print(f"  Matrix shape: {sim_df.shape}")
    return sim_df

def get_recommendations(model_name, df_model, sim_df, top_n=5,
                        same_segment_only=False):
    
    model_name     = model_name.strip()
    sim_df.index   = sim_df.index.str.strip()
    sim_df.columns = sim_df.columns.str.strip()

    if model_name not in sim_df.index:
        print(f"  '{model_name}' not found")
        return None
    
    selected_segment = df_model[df_model['model'] == model_name]['segment'].values[0]

    scores = sim_df[model_name].sort_values(ascending= False)
    top_models = scores.iloc[1:].index.tolist()

    result = df_model[df_model['model'].isin(top_models)].copy()
    result['similarity_score'] = result['model'].map( lambda m: round(scores[m], 3))

    if same_segment_only:
        same_seg = result[result['segment'] == selected_segment]

        if len(same_seg) >= top_n:
            result = same_seg

        else:
            current_idx = segment_order.index(selected_segment)
            nearby_segs = []
            if current_idx > 0:
                nearby_segs.append(segment_order[current_idx - 1])
            if current_idx < len(segment_order) - 1:
                nearby_segs.append(segment_order[current_idx + 1])

            nearby = result[result['segment'].isin(nearby_segs)]
            result  = pd.concat(
                [same_seg, nearby]).drop_duplicates()
            
    result = result.sort_values(by='similarity_score', ascending=False)
    result = result.head(top_n).reset_index(drop=True)
    result.index += 1

    return result[['brand', 'model', 'segment','price_usd', 'camera_rating', 'battery_life_rating',
                   'performance_rating', 'similarity_score']]









