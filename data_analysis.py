import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_crop_recommendations():
    """Perform EDA on Crop_recommendations dataset"""
    df = pd.read_csv('data/Crop_recommendations_cleaned.csv')
    
    print("\n=== CROP RECOMMENDATIONS ANALYSIS ===")
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print(f"\nCrop Distribution:\n{df['label'].value_counts()}")
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix - Crop Recommendations')
    plt.tight_layout()
    plt.savefig('data/crop_recommendations_correlation.png')
    plt.close()
    
    # Feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx//4, idx%4]
        df[col].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('data/crop_recommendations_distributions.png')
    plt.close()
    
    print("\nAnalysis plots saved!")

def analyze_data_core():
    """Perform EDA on data_core dataset"""
    df = pd.read_csv('data/data_core_cleaned.csv')
    
    print("\n=== DATA CORE ANALYSIS ===")
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print(f"\nCrop Type Distribution:\n{df['Crop Type'].value_counts()}")
    print(f"\nSoil Type Distribution:\n{df['Soil Type'].value_counts()}")
    print(f"\nFertilizer Distribution:\n{df['Fertilizer Name'].value_counts()}")
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix - Data Core')
    plt.tight_layout()
    plt.savefig('data/data_core_correlation.png')
    plt.close()
    
    # Feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx//3, idx%3]
        df[col].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('data/data_core_distributions.png')
    plt.close()
    
    print("\nAnalysis plots saved!")

if __name__ == "__main__":
    print("Performing data analysis...")
    analyze_crop_recommendations()
    analyze_data_core()
    print("\nData analysis completed!")
