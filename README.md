<<<<<<< HEAD

# Crop-Recommendation-System

It predicts the most suitable crop by analyzing factors like:  Soil nutrients (N, P, K)  Soil pH  Temperature  Rainfall  Humidity  Location-specific climate patterns  The goal is to maximize yield, reduce risk, and help farmers make data‑driven decisions
=======

# 🌾 Crop Recommendation System

> **Status**: ✅Completed

A comprehensive Machine Learning web application that helps farmers make informed decisions. This system recommends the most suitable crops, fertilizers, and predicts crop types based on soil and climate parameters using advanced Random Forest algorithms.

## ✨ Key Features

1. **🌱 Crop Recommendation**: Suggests the best crop to grow based on Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
2. **🧪 Fertilizer Recommendation**: Recommends the appropriate fertilizer based on soil composition and crop type.
3. **🔍 Crop Type Prediction**: Predicts the general crop type suitable for the given soil conditions.
4. **💻 Interactive Web Interface**: User-friendly Flask web application for real-time predictions.

## 🛠️ Tech Stack

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Pandas, NumPy
* **Frontend**: HTML, CSS, JavaScript
* **Visualization**: Matplotlib, Seaborn (for analysis)

## 📁 Project Structure

```
Crop/
│
├── app/                # Main Application Directory
│   └── app.py          # Flask Application Entry Point
│
├── data/               # Dataset Directory
│   ├── crop_recommendation.csv
│   └── data_core.csv
│
├── models/             # Trained ML Models
│   ├── crop_recommendation_model.pkl
│   ├── fertilizer_recommendation_model.pkl
│   └── crop_type_prediction_model.pkl
│
├── templates/          # HTML Templates
│   ├── index.html
│   ├── crop.html
│   ├── fertilizer.html
│   └── type.html
│
├── static/             # Static Assets (CSS/JS)
│
├── documentation/      # Project Documentation
│
├── run_pipeline.py     # Master script to retrain all models
├── train_models.py     # Model training logic
├── data_preprocessing.py # Data cleaning logic
├── data_analysis.py    # Exploratory Data Analysis
└── requirements.txt    # Project Dependencies
```

## 🚀 Installation & Setup

### 1. Clone or Download

# 🌾 Crop Recommendation System

> **Status**: ✅Completed

A comprehensive Machine Learning web application that helps farmers make informed decisions. This system recommends the most suitable crops, fertilizers, and predicts crop types based on soil and climate parameters using advanced Random Forest algorithms.

## ✨ Key Features

1. **🌱 Crop Recommendation**: Suggests the best crop to grow based on Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
2. **🧪 Fertilizer Recommendation**: Recommends the appropriate fertilizer based on soil composition and crop type.
3. **🔍 Crop Type Prediction**: Predicts the general crop type suitable for the given soil conditions.
4. **💻 Interactive Web Interface**: User-friendly Flask web application for real-time predictions.

## 🛠️ Tech Stack

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Pandas, NumPy
* **Frontend**: HTML, CSS, JavaScript
* **Visualization**: Matplotlib, Seaborn (for analysis)

## 📁 Project Structure

```
Crop/
│
├── app/                # Main Application Directory
│   └── app.py          # Flask Application Entry Point
│
├── data/               # Dataset Directory
│   ├── crop_recommendation.csv
│   └── data_core.csv
│
├── models/             # Trained ML Models
│   ├── crop_recommendation_model.pkl
│   ├── fertilizer_recommendation_model.pkl
│   └── crop_type_prediction_model.pkl
│
├── templates/          # HTML Templates
│   ├── index.html
│   ├── crop.html
│   ├── fertilizer.html
│   └── type.html
│
├── static/             # Static Assets (CSS/JS)
│
├── documentation/      # Project Documentation
│
├── run_pipeline.py     # Master script to retrain all models
├── train_models.py     # Model training logic
├── data_preprocessing.py # Data cleaning logic
├── data_analysis.py    # Exploratory Data Analysis
└── requirements.txt    # Project Dependencies
```

## 🚀 Installation & Setup

### 1. Clone or Download

Download the project files to your local machine.

### 2. Create a Virtual Environment

It's recommended to use a virtual environment.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

## 🌐 How to Run

1. **Start the Application**:

    ```bash
    python app.py
    ```

2. **Access the Web App**:
    Open your browser and go to: `http://127.0.0.1:5000`

## 🧠 Model Training (Optional)

If you want to retrain the models from scratch using the datasets:

Run the **Master Pipeline**:

```bash
python run_pipeline.py
```

This script will automatically:

1. Clean the data (`data_preprocessing.py`)
2. Analyze the data (`data_analysis.py`)
3. Train and save new models (`train_models.py`)

## 📊 Model Performance

The system uses **Random Forest Classifiers** for all predictions, achieving high accuracy:

* **Crop Recommendation**: ~99% Accuracy
* **Fertilizer Recommendation**: ~98% Accuracy
* **Crop Type Prediction**: ~98% Accuracy

## 📝 Dataset Info

The project uses two main datasets:

1. **Crop Recommendation**: 2200 samples, 7 features (N, P, K, Temp, Humidity, pH, Rain), 22 classes.
2. **Data Core**: Specialized dataset for fertilizer and crop type analysis.

---
*Developed by Atul*
