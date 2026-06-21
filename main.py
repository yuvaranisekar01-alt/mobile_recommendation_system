import os
import pandas as pd
from src.preprocessor import preprocess
from src.segmentation import segment
from src.recommender import build_similarity_matrix, get_recommendations

raw_data_path = "C:\projects\mobile_review_recommendation\data\Raw\Mobile Reviews Sentiment null.csv"
processed_dir =  "C:\projects\mobile_review_recommendation\data\Processed"

def run_pipeline():
    print("=" * 50)
    print("   MOBILE RECOMMENDATION PIPELINE")
    print("=" * 50)

    #load 
    df= pd.read_csv(raw_data_path)
    print(f"\n[1/4] Loaded {df.shape[0]} rows")

    # Preprocess
    print("\n[2/4] Preprocessing...")
    df_model = preprocess(df)

    # Segment
    print("\n[3/4] Segmenting...")
    df_model = segment(df_model)

    # Recommend
    print("\n[4/4] Building recommender...")
    sim_df = build_similarity_matrix(df_model)

    # Save outputs
    os.makedirs(processed_dir, exist_ok=True)
    df_model.to_csv(f'{processed_dir}/model_features.csv', index=False)
    sim_df.to_csv(  f'{processed_dir}/similarity_matrix.csv')
    print(f"\n  Saved to '{processed_dir}/'")
    print("  → model_features.csv")
    print("  → similarity_matrix.csv")

    print("\n  Pipeline complete!")
    print("  Run: streamlit run dashboard/app.py\n")

    return df_model, sim_df


if __name__ == "__main__":
    run_pipeline()


