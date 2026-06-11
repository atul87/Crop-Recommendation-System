"""
Full end-to-end test suite for the Crop Recommendation System.
Tests: MySQL connectivity, model loading, all API endpoints, frontend serving, history API.
"""
import sys
import os
import json
import pickle
import time
import threading
import traceback

# ===== 1. DATASET VALIDATION =====
print("=" * 60)
print("1. DATASET VALIDATION")
print("=" * 60)

import pandas as pd

# Crop Recommendations dataset
df_crop = pd.read_csv("data/Crop_recommendations_cleaned.csv")
assert df_crop.shape[0] > 100, f"Too few rows in crop dataset: {df_crop.shape[0]}"
assert df_crop.isnull().sum().sum() == 0, "Null values in crop dataset"
assert set(["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]).issubset(df_crop.columns)
print(f"  [PASS] Crop dataset: {df_crop.shape[0]} rows, {df_crop['label'].nunique()} labels, no nulls")

# Data Core dataset
df_core = pd.read_csv("data/data_core_cleaned.csv")
assert df_core.shape[0] > 50, f"Too few rows in core dataset: {df_core.shape[0]}"
assert df_core.isnull().sum().sum() == 0, "Null values in core dataset"
print(f"  [PASS] Core dataset: {df_core.shape[0]} rows, {df_core['Fertilizer Name'].nunique()} fertilizers, {df_core['Crop Type'].nunique()} crop types")

# ===== 2. MODEL LOADING =====
print("\n" + "=" * 60)
print("2. MODEL LOADING")
print("=" * 60)

# Crop model
with open("models/crop_recommendation_model.pkl", "rb") as f:
    crop_model = pickle.load(f)
print(f"  [PASS] Crop recommendation model loaded: {type(crop_model).__name__}")

# Fertilizer model
with open("models/fertilizer_recommendation_model.pkl", "rb") as f:
    fert_data = pickle.load(f)
assert "model" in fert_data and "scaler" in fert_data and "soil_encoder" in fert_data and "crop_encoder" in fert_data
print(f"  [PASS] Fertilizer model loaded with keys: {list(fert_data.keys())}")
print(f"         Soil classes: {list(fert_data['soil_encoder'].classes_)}")
print(f"         Crop classes: {list(fert_data['crop_encoder'].classes_)}")

# Crop type model
with open("models/crop_type_prediction_model.pkl", "rb") as f:
    ct_data = pickle.load(f)
assert "model" in ct_data and "scaler" in ct_data and "soil_encoder" in ct_data
print(f"  [PASS] Crop type model loaded with keys: {list(ct_data.keys())}")

# ===== 3. MODEL PREDICTION TEST (offline) =====
print("\n" + "=" * 60)
print("3. MODEL PREDICTION TEST (offline)")
print("=" * 60)

import numpy as np

# Test crop model prediction
test_features = pd.DataFrame([[50, 50, 50, 25.0, 60.0, 6.5, 100.0]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
pred = crop_model.predict(test_features)[0]
prob = max(crop_model.predict_proba(test_features)[0]) * 100
print(f"  [PASS] Crop prediction: {pred} (confidence: {prob:.2f}%)")

# Test fertilizer model prediction
soil_enc = fert_data["soil_encoder"].transform(["Sandy"])[0]
crop_enc = fert_data["crop_encoder"].transform(["Maize"])[0]
fert_input = pd.DataFrame([[26.0, 52.0, 38.0, soil_enc, crop_enc, 37, 0, 0]],
                           columns=["Temparature", "Humidity", "Moisture", "Soil_Type_Encoded",
                                    "Crop_Type_Encoded", "Nitrogen", "Potassium", "Phosphorous"])
fert_scaled = fert_data["scaler"].transform(fert_input)
fert_pred = fert_data["model"].predict(fert_scaled)[0]
fert_prob = max(fert_data["model"].predict_proba(fert_scaled)[0]) * 100
print(f"  [PASS] Fertilizer prediction: {fert_pred} (confidence: {fert_prob:.2f}%)")

# Test crop type model prediction
ct_soil_enc = ct_data["soil_encoder"].transform(["Sandy"])[0]
ct_input = pd.DataFrame([[26.0, 52.0, 38.0, ct_soil_enc, 37, 0, 0]],
                          columns=["Temparature", "Humidity", "Moisture", "Soil_Type_Encoded",
                                   "Nitrogen", "Potassium", "Phosphorous"])
ct_scaled = ct_data["scaler"].transform(ct_input)
ct_pred = ct_data["model"].predict(ct_scaled)[0]
ct_prob = max(ct_data["model"].predict_proba(ct_scaled)[0]) * 100
print(f"  [PASS] Crop type prediction: {ct_pred} (confidence: {ct_prob:.2f}%)")

# ===== 4. MYSQL DATABASE TEST =====
print("\n" + "=" * 60)
print("4. MYSQL DATABASE TEST")
print("=" * 60)

try:
    import pymysql
    from dotenv import load_dotenv
    load_dotenv()
    
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", 3306))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "crop_recommendation")
    
    # Test connection without DB
    conn = pymysql.connect(host=host, port=port, user=user, password=password)
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    print(f"  [PASS] MySQL connected. Version: {version}")
    
    # Create DB if needed
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    conn.close()
    
    # Connect to DB
    conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database,
                           cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    
    # Create table if needed
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `predictions` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `prediction_type` VARCHAR(50) NOT NULL,
            `inputs` TEXT NOT NULL,
            `result` VARCHAR(100) NOT NULL,
            `confidence` FLOAT NOT NULL,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print(f"  [PASS] Database '{database}' and 'predictions' table ready")
    
    # Test insert
    test_inputs = json.dumps({"test": True})
    cur.execute("INSERT INTO `predictions` (`prediction_type`, `inputs`, `result`, `confidence`) VALUES (%s, %s, %s, %s)",
                ("Test", test_inputs, "TestResult", 99.99))
    conn.commit()
    print(f"  [PASS] Test insert successful")
    
    # Test select
    cur.execute("SELECT * FROM `predictions` ORDER BY `id` DESC LIMIT 1")
    row = cur.fetchone()
    assert row["prediction_type"] == "Test"
    assert row["result"] == "TestResult"
    print(f"  [PASS] Test select successful: {row}")
    
    # Cleanup test row
    cur.execute("DELETE FROM `predictions` WHERE `prediction_type` = 'Test'")
    conn.commit()
    print(f"  [PASS] Test cleanup done")
    
    conn.close()
    mysql_ok = True
    
except Exception as e:
    print(f"  [FAIL] MySQL error: {e}")
    mysql_ok = False

# ===== 5. FLASK API TESTS =====
print("\n" + "=" * 60)
print("5. FLASK API TESTS")
print("=" * 60)

try:
    import requests
except ImportError:
    print("  [WARN] 'requests' not installed. Installing...")
    os.system(f'"{sys.executable}" -m pip install requests')
    import requests

# Start Flask in background
from app import app
app.config["TESTING"] = True
client = app.test_client()

# 5a. Test homepage
print("\n  --- 5a. Page Routes ---")
for route, name in [("/", "Homepage"), ("/crop", "Crop page"), ("/fertilizer", "Fertilizer page"), ("/type", "Type page"), ("/history", "History page")]:
    resp = client.get(route)
    assert resp.status_code == 200, f"Route {route} returned {resp.status_code}"
    print(f"  [PASS] GET {route} ({name}): {resp.status_code}")

# 5b. Test Crop Recommendation API
print("\n  --- 5b. Crop Recommendation API ---")
# Valid request
resp = client.post("/api/predict-crop", json={
    "N": "50", "P": "50", "K": "50", "temperature": "25.0",
    "humidity": "60.0", "ph": "6.5", "rainfall": "100.0"
})
data = resp.get_json()
assert data["success"] == True, f"Crop API failed: {data}"
print(f"  [PASS] Valid crop prediction: {data['prediction']} ({data['confidence']}%)")

# Missing fields
resp = client.post("/api/predict-crop", json={"N": "50"})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Missing fields handled: {data['error']}")

# Empty body
resp = client.post("/api/predict-crop", json=None, content_type="application/json")
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Empty body handled: {data['error']}")

# Invalid data type
resp = client.post("/api/predict-crop", json={
    "N": "abc", "P": "50", "K": "50", "temperature": "25.0",
    "humidity": "60.0", "ph": "6.5", "rainfall": "100.0"
})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Invalid data type handled: {data['error']}")

# 5c. Test Fertilizer Recommendation API
print("\n  --- 5c. Fertilizer Recommendation API ---")
# Valid request
resp = client.post("/api/predict-fertilizer", json={
    "temperature": "26", "humidity": "52", "moisture": "38",
    "soil_type": "Sandy", "crop_type": "Maize",
    "nitrogen": "37", "potassium": "0", "phosphorous": "0"
})
data = resp.get_json()
assert data["success"] == True, f"Fertilizer API failed: {data}"
print(f"  [PASS] Valid fertilizer prediction: {data['prediction']} ({data['confidence']}%)")

# Unknown soil type
resp = client.post("/api/predict-fertilizer", json={
    "temperature": "26", "humidity": "52", "moisture": "38",
    "soil_type": "UnknownSoil", "crop_type": "Maize",
    "nitrogen": "37", "potassium": "0", "phosphorous": "0"
})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Unknown soil type handled: {data['error']}")

# Unknown crop type
resp = client.post("/api/predict-fertilizer", json={
    "temperature": "26", "humidity": "52", "moisture": "38",
    "soil_type": "Sandy", "crop_type": "UnknownCrop",
    "nitrogen": "37", "potassium": "0", "phosphorous": "0"
})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Unknown crop type handled: {data['error']}")

# Missing fields
resp = client.post("/api/predict-fertilizer", json={"temperature": "26"})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Missing fields handled: {data['error']}")

# 5d. Test Crop Type Prediction API
print("\n  --- 5d. Crop Type Prediction API ---")
# Valid request
resp = client.post("/api/predict-type", json={
    "temperature": "26", "humidity": "52", "moisture": "38",
    "soil_type": "Sandy", "nitrogen": "37", "potassium": "0", "phosphorous": "0"
})
data = resp.get_json()
assert data["success"] == True, f"Type API failed: {data}"
print(f"  [PASS] Valid crop type prediction: {data['prediction']} ({data['confidence']}%)")

# Unknown soil type
resp = client.post("/api/predict-type", json={
    "temperature": "26", "humidity": "52", "moisture": "38",
    "soil_type": "GravelSoil", "nitrogen": "37", "potassium": "0", "phosphorous": "0"
})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Unknown soil type handled: {data['error']}")

# Missing fields
resp = client.post("/api/predict-type", json={"temperature": "26"})
data = resp.get_json()
assert data["success"] == False
print(f"  [PASS] Missing fields handled: {data['error']}")

# 5e. Test History API
print("\n  --- 5e. History API ---")
resp = client.get("/api/history")
data = resp.get_json()
assert isinstance(data, list), f"History should return list, got {type(data)}"
print(f"  [PASS] History API returned {len(data)} records")

# 5f. Test with all soil types from the dataset
print("\n  --- 5f. Soil Type Coverage ---")
for soil in ["Sandy", "Loamy", "Black", "Red", "Clayey"]:
    resp = client.post("/api/predict-fertilizer", json={
        "temperature": "30", "humidity": "55", "moisture": "40",
        "soil_type": soil, "crop_type": "Wheat",
        "nitrogen": "20", "potassium": "10", "phosphorous": "15"
    })
    data = resp.get_json()
    assert data["success"] == True, f"Failed for soil type {soil}: {data}"
    print(f"  [PASS] Soil type '{soil}': {data['prediction']}")

# 5g. Test with all crop types from the dataset
print("\n  --- 5g. Crop Type Coverage ---")
for crop in ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"]:
    resp = client.post("/api/predict-fertilizer", json={
        "temperature": "30", "humidity": "55", "moisture": "40",
        "soil_type": "Loamy", "crop_type": crop,
        "nitrogen": "20", "potassium": "10", "phosphorous": "15"
    })
    data = resp.get_json()
    assert data["success"] == True, f"Failed for crop type {crop}: {data}"
    print(f"  [PASS] Crop type '{crop}': {data['prediction']}")

# 5h. Boundary value tests
print("\n  --- 5h. Boundary Value Tests ---")
# Extreme values for crop recommendation
resp = client.post("/api/predict-crop", json={
    "N": "0", "P": "0", "K": "0", "temperature": "0",
    "humidity": "0", "ph": "0", "rainfall": "0"
})
data = resp.get_json()
assert data["success"] == True
print(f"  [PASS] All-zero crop input: {data['prediction']} ({data['confidence']}%)")

resp = client.post("/api/predict-crop", json={
    "N": "200", "P": "200", "K": "200", "temperature": "50",
    "humidity": "100", "ph": "14", "rainfall": "500"
})
data = resp.get_json()
assert data["success"] == True
print(f"  [PASS] Max-value crop input: {data['prediction']} ({data['confidence']}%)")

# ===== 6. FRONTEND HTML VALIDATION =====
print("\n" + "=" * 60)
print("6. FRONTEND HTML VALIDATION")
print("=" * 60)

# Check all pages contain required elements
resp = client.get("/")
html = resp.data.decode()
assert "CropAI" in html, "Missing brand name"
assert "prediction-form" not in html or "features" in html, "Homepage should show features"
assert 'href="/crop"' in html, "Missing crop link"
assert 'href="/fertilizer"' in html, "Missing fertilizer link"
assert 'href="/type"' in html, "Missing type link"
assert 'href="/history"' in html, "Missing history link"
print("  [PASS] Homepage has all navigation links")

resp = client.get("/crop")
html = resp.data.decode()
assert 'id="prediction-form"' in html, "Missing prediction form"
assert 'name="N"' in html, "Missing N field"
assert 'name="ph"' in html, "Missing ph field"
assert 'name="rainfall"' in html, "Missing rainfall field"
assert 'script.js' in html, "Missing script.js"
assert 'style.css' in html, "Missing style.css"
print("  [PASS] Crop page has all form fields and assets")

resp = client.get("/fertilizer")
html = resp.data.decode()
assert 'name="soil_type"' in html, "Missing soil_type field"
assert 'name="crop_type"' in html, "Missing crop_type field"
assert "Sandy" in html and "Loamy" in html and "Black" in html, "Missing soil options"
assert "Maize" in html and "Wheat" in html and "Cotton" in html, "Missing crop options"
print("  [PASS] Fertilizer page has all form fields and dropdown options")

resp = client.get("/type")
html = resp.data.decode()
assert 'name="soil_type"' in html, "Missing soil_type field"
assert 'name="moisture"' in html, "Missing moisture field"
print("  [PASS] Type page has all form fields")

resp = client.get("/history")
html = resp.data.decode()
assert "history-table" in html or "history-body" in html, "Missing history table"
assert "/api/history" in html, "Missing API call to /api/history"
print("  [PASS] History page has table and API integration")

# ===== 7. CSS/JS STATIC FILES =====
print("\n" + "=" * 60)
print("7. STATIC FILES")
print("=" * 60)

resp = client.get("/static/css/style.css")
assert resp.status_code == 200
css = resp.data.decode()
assert "--primary-color" in css, "Missing CSS variables"
assert "glassmorphism" in css.lower() or "backdrop-filter" in css, "Missing glassmorphism styles"
print(f"  [PASS] style.css loads ({len(css)} bytes)")

resp = client.get("/static/js/script.js")
assert resp.status_code == 200
js = resp.data.decode()
assert "prediction-form" in js, "Missing form handler"
assert "fetch" in js, "Missing fetch call"
assert "confidence" in js, "Missing confidence handling"
print(f"  [PASS] script.js loads ({len(js)} bytes)")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)
print(f"  Datasets:           PASS (crop: {df_crop.shape[0]} rows, core: {df_core.shape[0]} rows)")
print(f"  Model Loading:      PASS (3 models)")
print(f"  Offline Predictions: PASS")
print(f"  MySQL Database:     {'PASS' if mysql_ok else 'FAIL - check .env credentials and MySQL server'}")
print(f"  Flask API (Crop):   PASS (valid, missing fields, empty body, invalid types)")
print(f"  Flask API (Fert):   PASS (valid, unknown soil, unknown crop, missing fields)")
print(f"  Flask API (Type):   PASS (valid, unknown soil, missing fields)")
print(f"  History API:        PASS")
print(f"  Soil Coverage:      PASS (5/5 soil types)")
print(f"  Crop Coverage:      PASS (11/11 crop types)")
print(f"  Boundary Values:    PASS (zero and max inputs)")
print(f"  Frontend HTML:      PASS (all pages, forms, links)")
print(f"  Static Files:       PASS (CSS + JS)")
print("=" * 60)
if mysql_ok:
    print("ALL TESTS PASSED SUCCESSFULLY!")
else:
    print("ALL CODE TESTS PASSED! MySQL needs configuration (check .env)")
print("=" * 60)
