import pandas as pd
import numpy as np

drop_cols = [ 'review_id', 'customer_name', 'age', 'exchange_rate_to_usd',
    'price_local', 'currency', 'language',
    'review_date', 'helpful_votes' ]

rating_cols = [ 'battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating',
       'display_rating', 'rating' ]

def preprocess(df):
    print("[Preprocessor] Starting...")
    df = df.copy()

    #fix capitalisation
    df['brand'] = df['brand'].str.strip().str.title()
    df['model'] = df['model'].str.strip().str.title()

    #drop irrelevant cols
    df= df.drop(columns= drop_cols, axis=1)

    #handle missing values
    df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())
    for col in rating_cols:
        df[col]= df[col].fillna(df[col].median())

    # aggregating columns
    agg_dict = {'price_usd': 'mean', 'battery_life_rating': 'mean', 'camera_rating' : 'mean',
                'performance_rating': 'mean', 'design_rating': 'mean', 'display_rating': 'mean',
                'rating': 'mean'}
    df_model = df.groupby(['brand', 'model']).agg(agg_dict).reset_index()
    df_model = df_model.round(2)

    print(f"  Done → {df_model.shape[0]} unique models")
    return df_model



