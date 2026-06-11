# 🌾 Crop Recommendation System (CropAI)

A comprehensive machine learning web application designed to help farmers and agronomists make data-driven agricultural decisions. The system recommends the most suitable crops, suggests appropriate fertilizers, and predicts crop types based on soil composition and environmental/climatic parameters.

The application features a modern **Tech-Agriculture glassmorphic user interface**, robust input validations, MySQL-backed prediction history, and optimized machine learning pipelines.

---

## ✨ Features

1.  **🌱 Crop Recommendation**
    *   Recommends the best crop to cultivate based on Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall metrics.
2.  **🧪 Fertilizer Recommendation**
    *   Suggests the correct fertilizer type based on soil chemistry, environmental conditions, and the specific crop type.
3.  **🔍 Crop Type Prediction**
    *   Classifies and predicts the category of crop suitable for given soil and environmental conditions.
4.  **📊 Prediction History Dashboard**
    *   All prediction results are stored in a MySQL database and can be viewed on the History page with type badges, input details, and confidence scores.
5.  **💻 Glassmorphic UI Dashboard**
    *   Premium dark-themed user interface featuring micro-animations, glowing fields, button loading states, and dynamic animated confidence indicators.
6.  **⚙️ Robust API Enforcements**
    *   Protects APIs against invalid inputs, missing fields, unseen categories (e.g. unknown soil type), and malformed request bodies by returning clear, structured JSON error payloads instead of crashing the server.

---

## 🛠️ Tech Stack

*   **Backend:** Python, Flask
*   **Database:** MySQL (via PyMySQL) with automatic schema initialization
*   **Machine Learning:** Scikit-learn (Random Forest Classifiers with GridSearchCV), Pandas, NumPy, StandardScaler, LabelEncoder
*   **Frontend:** Modern HTML5, CSS3 (Glassmorphism & HSL gradients), Vanilla JavaScript
*   **Visualization:** Matplotlib, Seaborn
*   **Testing:** Python integration test suite (test_full.py)

---

## 📁 Project Structure

```
Crop/
│
├── app.py                # Main Flask Application Entry Point
├── database.py           # MySQL database connection, schema init, and CRUD operations
├── run_pipeline.py       # Master script to run data cleaning, EDA, and model training
├── train_models.py       # Model training with GridSearchCV & Cross-Validation
├── data_preprocessing.py # Data cleaning, proper outlier masking, and data slicing
├── data_analysis.py      # Exploratory Data Analysis & plot generation
├── test_full.py          # Comprehensive end-to-end test suite
├── test_endpoints.py     # Legacy API integration tests
├── requirements.txt      # Python Dependencies
├── .env                  # MySQL connection configuration
│
├── data/                 # Raw and Cleaned Datasets
│   ├── crop_recommendation.csv
│   ├── Crop_recommendations_cleaned.csv  (1768 rows, 20 crop labels)
│   ├── data_core.csv
│   └── data_core_cleaned.csv             (99 rows, 7 fertilizers, 11 crop types)
│
├── models/               # Saved ML Models & Performance Plots
│   ├── crop_recommendation_model.pkl
│   ├── fertilizer_recommendation_model.pkl
│   ├── crop_type_prediction_model.pkl
│   └── *_confusion_matrix.png
│
├── static/               # Frontend Static Assets
│   ├── css/
│   │   └── style.css     # Glassmorphic AgriTech stylesheet
│   └── js/
│       └── script.js     # Input validator & confidence bar animator
│
└── templates/            # Jinja HTML Templates
    ├── index.html        # Smart Farming Solutions homepage
    ├── crop.html         # Crop Recommendation tool
    ├── fertilizer.html   # Fertilizer Recommendation tool
    ├── type.html         # Crop Type Prediction tool
    └── history.html      # MySQL-backed Prediction History dashboard
```

---

## 🧠 Model Pipeline & Performance

### 1. Data Cleaning (`data_preprocessing.py`)
*   **Outlier Removal Bug Fix:** Replaced sequential in-loop DataFrame filtering (which dynamically shifted column distributions) with a static boolean mask, making outlier removal order-independent.
*   **Slicing Synthetic Noise:** Cleaned `data_core.csv` by slicing it to the first 99 genuine rows. The remaining 7,901 rows were noisy generated data with shuffled labels that previously corrupted training.

### 2. ML Training (`train_models.py`)
*   Random Forest models tuned using `GridSearchCV` (n_estimators: [50, 100, 200], max_depth: [None, 10, 20]).
*   Scored using 5-fold cross-validation.
*   Feature importances and confusion matrices saved to `models/`.

### 3. Model Accuracy Comparison

| Prediction Model | Previous Accuracy (Noisy Data) | Updated Accuracy (Cleaned Data) | Change |
| :--- | :---: | :---: | :---: |
| **Crop Recommendation** | 99.19% | **98.87%** | -0.32% (Stabilized) |
| **Fertilizer Recommendation** | 16.08% | **95.00%** | **+78.92% (Restored)** |
| **Crop Type Prediction** | 9.59% | **65.00%** | **+55.41% (Restored)** |

---

## 🚀 Installation & Setup

### 1. Clone or Download
```bash
git clone <repository-url>
cd Crop
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure MySQL Database
Edit the `.env` file with your MySQL credentials:
```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=crop_recommendation
```
> **Note:** The database and tables are created automatically when the app starts. Just make sure MySQL is running and the credentials are correct.

---

## 🌐 Running & Verification

### 1. Run the ML Pipeline (Optional)
To retrain all models from scratch:
```bash
python run_pipeline.py
```

### 2. Run End-to-End Tests
Comprehensive test suite covering datasets, models, APIs, frontend, and database:
```bash
python test_full.py
```
*Expected output: `ALL TESTS PASSED SUCCESSFULLY!`*

### 3. Run the Web Application
```bash
python app.py
```
Access the application at: **http://127.0.0.1:5000**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| GET | `/` | Homepage |
| GET | `/crop` | Crop Recommendation form |
| GET | `/fertilizer` | Fertilizer Recommendation form |
| GET | `/type` | Crop Type Prediction form |
| GET | `/history` | Prediction History dashboard |
| POST | `/api/predict-crop` | Predict best crop (JSON body) |
| POST | `/api/predict-fertilizer` | Recommend fertilizer (JSON body) |
| POST | `/api/predict-type` | Predict crop type (JSON body) |
| GET | `/api/history` | Get all prediction records (JSON) |

### Example API Request (Crop Recommendation)
```json
POST /api/predict-crop
{
    "N": "50", "P": "50", "K": "50",
    "temperature": "25.0", "humidity": "60.0",
    "ph": "6.5", "rainfall": "100.0"
}
```

### Example API Response
```json
{
    "success": true,
    "prediction": "mango",
    "confidence": "35.00"
}
```

---

*Developed by Atul • Enhanced with Premium AgriTech Engineering*
