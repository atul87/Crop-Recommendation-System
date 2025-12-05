# 📊 Datasets Documentation

## Overview
This project uses two main datasets for crop and fertilizer recommendations:

### 1. Crop_recommendations.csv
**Purpose:** Predict the most suitable crop based on soil and climate parameters

**Features:**
- `N` - Nitrogen content in soil (kg/ha)
- `P` - Phosphorus content in soil (kg/ha)
- `K` - Potassium content in soil (kg/ha)
- `temperature` - Temperature in Celsius
- `humidity` - Relative humidity in %
- `ph` - pH value of soil
- `rainfall` - Rainfall in mm
- `label` - Target crop (22 different crops)

**Crops:** rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

**Size:** 2200 samples

---

### 2. data_core.csv
**Purpose:** Predict fertilizer recommendations and crop types based on soil conditions

**Features:**
- `Temparature` - Temperature in Celsius
- `Humidity` - Relative humidity in %
- `Moisture` - Soil moisture level
- `Soil Type` - Type of soil (Sandy, Loamy, Black, Red, Clayey)
- `Crop Type` - Type of crop being grown
- `Nitrogen` - Nitrogen content
- `Potassium` - Potassium content
- `Phosphorous` - Phosphorous content
- `Fertilizer Name` - Recommended fertilizer (Target variable)

**Fertilizers:** Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26

**Crops:** Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts

**Size:** ~200 samples

---

## 🔄 Data Processing Pipeline

### Step 1: Data Cleaning (`data_preprocessing.py`)
- Removes duplicate records
- Handles missing values
- Removes outliers using IQR method
- Generates cleaned datasets

### Step 2: Data Analysis (`data_analysis.py`)
- Statistical summaries
- Distribution analysis
- Correlation matrices
- Feature importance visualization

### Step 3: Model Training (`train_models.py`)
Three models are trained:
1. **Crop Recommendation Model** - Predicts best crop for given conditions
2. **Fertilizer Recommendation Model** - Suggests optimal fertilizer
3. **Crop Type Prediction Model** - Predicts suitable crop type

---

## 🚀 Quick Start

### Run Complete Pipeline
```bash
python run_pipeline.py
```

This will:
1. Clean both datasets
2. Perform exploratory data analysis
3. Train all three ML models
4. Generate visualizations and reports

### Individual Steps

**Clean Data:**
```bash
python data_preprocessing.py
```

**Analyze Data:**
```bash
python data_analysis.py
```

**Train Models:**
```bash
python train_models.py
```

**Make Predictions:**
```bash
python predict.py
```

---

## 📈 Model Performance

### Expected Accuracy
- Crop Recommendation: ~97-99%
- Fertilizer Recommendation: ~85-95%
- Crop Type Prediction: ~90-95%

### Models Used
- Algorithm: Random Forest Classifier
- Features: Scaled and encoded appropriately
- Validation: 80-20 train-test split

---

## 📁 Generated Files

After running the pipeline:

**Cleaned Data:**
- `data/Crop_recommendations_cleaned.csv`
- `data/data_core_cleaned.csv`

**Visualizations:**
- `data/crop_recommendations_correlation.png`
- `data/crop_recommendations_distributions.png`
- `data/data_core_correlation.png`
- `data/data_core_distributions.png`

**Models:**
- `models/crop_recommendation_model.pkl`
- `models/fertilizer_recommendation_model.pkl`
- `models/crop_type_prediction_model.pkl`

**Confusion Matrices:**
- `models/crop_recommendation_confusion_matrix.png`
- `models/fertilizer_recommendation_confusion_matrix.png`
- `models/crop_type_prediction_confusion_matrix.png`

---

## 💡 Usage Examples

### Predict Crop
```python
from predict import predict_crop_recommendation

crop, confidence = predict_crop_recommendation(
    N=90, P=42, K=43, 
    temperature=20.8, humidity=82.0, 
    ph=6.5, rainfall=202.9
)
print(f"Recommended: {crop} ({confidence:.2f}% confidence)")
```

### Predict Fertilizer
```python
from predict import predict_fertilizer

fertilizer, confidence = predict_fertilizer(
    temperature=26, humidity=52, moisture=38,
    soil_type='Sandy', crop_type='Maize',
    nitrogen=37, potassium=0, phosphorous=0
)
print(f"Recommended: {fertilizer} ({confidence:.2f}% confidence)")
```

### Predict Crop Type
```python
from predict import predict_crop_type

crop_type, confidence = predict_crop_type(
    temperature=26, humidity=52, moisture=38,
    soil_type='Sandy',
    nitrogen=37, potassium=0, phosphorous=0
)
print(f"Predicted: {crop_type} ({confidence:.2f}% confidence)")
```

---

## 🔍 Data Insights

### Crop_recommendations Dataset
- Most crops require moderate NPK levels
- Temperature range: 8-43°C
- Humidity range: 14-99%
- pH range: 3.5-9.9
- Rainfall range: 20-298mm

### data_core Dataset
- Temperature range: 10-43°C
- Humidity range: 43-99%
- Moisture range: 20-70%
- 5 soil types with different characteristics
- 11 major crop types

---

## 📊 Feature Importance

### Top Features for Crop Recommendation:
1. Rainfall
2. Potassium (K)
3. Nitrogen (N)
4. Temperature
5. Humidity

### Top Features for Fertilizer Recommendation:
1. Nitrogen content
2. Phosphorous content
3. Crop type
4. Soil type
5. Moisture

---

## 🛠️ Troubleshooting

**Issue:** Missing data files
- **Solution:** Ensure `Crop_recommendations.csv` and `data_core.csv` are in the `data/` folder

**Issue:** Model not found
- **Solution:** Run `python train_models.py` first

**Issue:** Import errors
- **Solution:** Install dependencies: `pip install -r requirements.txt`

---

## 📝 Notes

- All models use Random Forest for robust predictions
- Feature scaling is applied where necessary
- Categorical variables are label-encoded
- Models are saved using pickle for easy deployment
- Confusion matrices help visualize model performance

---

## 🎯 Next Steps

1. Run the pipeline: `python run_pipeline.py`
2. Check generated visualizations in `data/` and `models/`
3. Test predictions using `predict.py`
4. Integrate models into web app or API
5. Fine-tune hyperparameters for better accuracy

---

**Happy Farming! 🌾**
