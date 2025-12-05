import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_crop_recommendation_model():
    """Train model for Crop Recommendations dataset"""
    print("\n=== TRAINING CROP RECOMMENDATION MODEL ===")
    
    # Load cleaned data
    df = pd.read_csv('data/Crop_recommendations_cleaned.csv')
    
    # Prepare features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importance:\n{feature_importance}")
    
    # Save model
    with open('models/crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Crop Recommendation')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/crop_recommendation_confusion_matrix.png')
    plt.close()
    
    print("\nModel saved to models/crop_recommendation_model.pkl")
    return model, accuracy

def train_fertilizer_recommendation_model():
    """Train model for Fertilizer Recommendations (data_core)"""
    print("\n=== TRAINING FERTILIZER RECOMMENDATION MODEL ===")
    
    # Load cleaned data
    df = pd.read_csv('data/data_core_cleaned.csv')
    
    # Encode categorical variables
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    
    df['Soil_Type_Encoded'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop_Type_Encoded'] = le_crop.fit_transform(df['Crop Type'])
    
    # Prepare features and target
    X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type_Encoded', 
            'Crop_Type_Encoded', 'Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['Fertilizer Name']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importance:\n{feature_importance}")
    
    # Save model and encoders
    with open('models/fertilizer_recommendation_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'soil_encoder': le_soil,
            'crop_encoder': le_crop
        }, f)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Fertilizer Recommendation')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/fertilizer_recommendation_confusion_matrix.png')
    plt.close()
    
    print("\nModel saved to models/fertilizer_recommendation_model.pkl")
    return model, accuracy

def train_crop_prediction_model():
    """Train model for Crop Type prediction (data_core)"""
    print("\n=== TRAINING CROP TYPE PREDICTION MODEL ===")
    
    # Load cleaned data
    df = pd.read_csv('data/data_core_cleaned.csv')
    
    # Encode categorical variables
    le_soil = LabelEncoder()
    df['Soil_Type_Encoded'] = le_soil.fit_transform(df['Soil Type'])
    
    # Prepare features and target
    X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type_Encoded', 
            'Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['Crop Type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importance:\n{feature_importance}")
    
    # Save model and encoders
    with open('models/crop_type_prediction_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'soil_encoder': le_soil
        }, f)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title('Confusion Matrix - Crop Type Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/crop_type_prediction_confusion_matrix.png')
    plt.close()
    
    print("\nModel saved to models/crop_type_prediction_model.pkl")
    return model, accuracy

if __name__ == "__main__":
    print("Training ML models...")
    
    # Train all models
    model1, acc1 = train_crop_recommendation_model()
    model2, acc2 = train_fertilizer_recommendation_model()
    model3, acc3 = train_crop_prediction_model()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Crop Recommendation Model Accuracy: {acc1*100:.2f}%")
    print(f"Fertilizer Recommendation Model Accuracy: {acc2*100:.2f}%")
    print(f"Crop Type Prediction Model Accuracy: {acc3*100:.2f}%")
    print("="*60)
    print("\nAll models trained and saved successfully!")
