import pandas as pd
import numpy as np

def clean_crop_recommendations():
    """Clean Crop_recommendations.csv dataset"""
    df = pd.read_csv('data/crop_recommendation.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove outliers using IQR method properly with a single boolean mask
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = mask & (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
    df = df[mask]
    
    # Save cleaned data
    df.to_csv('data/Crop_recommendations_cleaned.csv', index=False)
    print(f"Crop_recommendations cleaned: {len(df)} rows")
    return df

def clean_data_core():
    """Clean data_core.csv dataset"""
    df = pd.read_csv('data/data_core.csv')
    
    # Slice to first 99 rows (genuine dataset)
    # The rows 100+ are noisy synthetic rows that break correlation/accuracy.
    df = df.iloc[:99].copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Save cleaned data
    df.to_csv('data/data_core_cleaned.csv', index=False)
    print(f"data_core cleaned: {len(df)} rows")
    return df

if __name__ == "__main__":
    print("Cleaning datasets...")
    df1 = clean_crop_recommendations()
    df2 = clean_data_core()
    print("Data cleaning completed!")
