import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_crop_recommendations():
    """Clean Crop_recommendations.csv dataset"""
    df = pd.read_csv('data/crop_recommendation.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    # Save cleaned data
    df.to_csv('data/Crop_recommendations_cleaned.csv', index=False)
    print(f"Crop_recommendations cleaned: {len(df)} rows")
    return df

def clean_data_core():
    """Clean data_core.csv dataset"""
    df = pd.read_csv('data/data_core.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove rows with missing Fertilizer Name
    df = df[df['Fertilizer Name'].notna()]
    
    # Remove outliers
    numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    # Save cleaned data
    df.to_csv('data/data_core_cleaned.csv', index=False)
    print(f"data_core cleaned: {len(df)} rows")
    return df

if __name__ == "__main__":
    print("Cleaning datasets...")
    df1 = clean_crop_recommendations()
    df2 = clean_data_core()
    print("Data cleaning completed!")
