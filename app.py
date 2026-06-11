from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import pickle
import database

app = Flask(__name__)

# Initialize database
database.init_db()

# Load Models
MODEL_DIR = "models"
try:
    # Crop Recommendation Model (Direct model)
    with open(os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"), "rb") as f:
        crop_model = pickle.load(f)

    # Fertilizer Recommendation Model (Dictionary with model, scaler, encoders)
    with open(
        os.path.join(MODEL_DIR, "fertilizer_recommendation_model.pkl"), "rb"
    ) as f:
        fertilizer_data = pickle.load(f)
        fertilizer_model = fertilizer_data["model"]
        fertilizer_scaler = fertilizer_data["scaler"]
        fertilizer_soil_encoder = fertilizer_data["soil_encoder"]
        fertilizer_crop_encoder = fertilizer_data["crop_encoder"]

    # Crop Type Prediction Model (Dictionary with model, scaler, encoder)
    with open(os.path.join(MODEL_DIR, "crop_type_prediction_model.pkl"), "rb") as f:
        crop_type_data = pickle.load(f)
        crop_type_model_obj = crop_type_data["model"]
        crop_type_scaler = crop_type_data["scaler"]
        crop_type_soil_encoder = crop_type_data["soil_encoder"]

    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    crop_model = None
    fertilizer_model = None
    crop_type_model_obj = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/crop")
def crop_page():
    return render_template("crop.html")


@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")


@app.route("/type")
def type_page():
    return render_template("type.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/api/history")
def get_history():
    records = database.get_predictions_history()
    return jsonify(records)


@app.route("/api/predict-crop", methods=["POST"])
def predict_crop():
    if crop_model is None:
        return jsonify({"success": False, "error": "Model not loaded on the server."})
    try:
        data = request.get_json(silent=True)
        if not data:
             return jsonify({"success": False, "error": "No input data provided."})
             
        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing_fields = [f for f in required_fields if f not in data or data[f] is None or data[f] == ""]
        if missing_fields:
             return jsonify({"success": False, "error": f"Missing or empty required fields: {', '.join(missing_fields)}"})

        features = pd.DataFrame(
            [
                [
                    float(data["N"]),
                    float(data["P"]),
                    float(data["K"]),
                    float(data["temperature"]),
                    float(data["humidity"]),
                    float(data["ph"]),
                    float(data["rainfall"]),
                ]
            ],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )

        prediction = crop_model.predict(features)[0]
        probabilities = crop_model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        confidence_str = f"{confidence:.2f}"

        # Save to database
        database.save_prediction("Crop Recommendation", data, prediction, confidence_str)

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "confidence": confidence_str,
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": "Invalid data type. Please provide numeric values."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None:
        return jsonify({"success": False, "error": "Model not loaded on the server."})
    try:
        data = request.get_json(silent=True)
        if not data:
             return jsonify({"success": False, "error": "No input data provided."})

        required_fields = ["temperature", "humidity", "moisture", "soil_type", "crop_type", "nitrogen", "potassium", "phosphorous"]
        missing_fields = [f for f in required_fields if f not in data or data[f] is None or data[f] == ""]
        if missing_fields:
             return jsonify({"success": False, "error": f"Missing or empty required fields: {', '.join(missing_fields)}"})

        # Encode categorical variables with error handling
        try:
            soil_type = data["soil_type"]
            soil_encoded = fertilizer_soil_encoder.transform([soil_type])[0]
        except ValueError:
            return jsonify({"success": False, "error": f"Unknown soil type: {soil_type}"})
            
        try:
            crop_type = data["crop_type"]
            crop_encoded = fertilizer_crop_encoder.transform([crop_type])[0]
        except ValueError:
            return jsonify({"success": False, "error": f"Unknown crop type: {crop_type}"})

        input_data = pd.DataFrame(
            [
                [
                    float(data["temperature"]),
                    float(data["humidity"]),
                    float(data["moisture"]),
                    soil_encoded,
                    crop_encoded,
                    float(data["nitrogen"]),
                    float(data["potassium"]),
                    float(data["phosphorous"]),
                ]
            ],
            columns=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Soil_Type_Encoded",
                "Crop_Type_Encoded",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
            ],
        )

        input_scaled = fertilizer_scaler.transform(input_data)
        prediction = fertilizer_model.predict(input_scaled)[0]
        probabilities = fertilizer_model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        confidence_str = f"{confidence:.2f}"

        # Save to database
        database.save_prediction("Fertilizer Recommendation", data, prediction, confidence_str)

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "confidence": confidence_str,
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": "Invalid data type. Please provide numeric values."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict-type", methods=["POST"])
def predict_type():
    if crop_type_model_obj is None:
        return jsonify({"success": False, "error": "Model not loaded on the server."})
    try:
        data = request.get_json(silent=True)
        if not data:
             return jsonify({"success": False, "error": "No input data provided."})

        required_fields = ["temperature", "humidity", "moisture", "soil_type", "nitrogen", "potassium", "phosphorous"]
        missing_fields = [f for f in required_fields if f not in data or data[f] is None or data[f] == ""]
        if missing_fields:
             return jsonify({"success": False, "error": f"Missing or empty required fields: {', '.join(missing_fields)}"})

        # Encode categorical variables with error handling
        try:
            soil_type = data["soil_type"]
            soil_encoded = crop_type_soil_encoder.transform([soil_type])[0]
        except ValueError:
            return jsonify({"success": False, "error": f"Unknown soil type: {soil_type}"})

        input_data = pd.DataFrame(
            [
                [
                    float(data["temperature"]),
                    float(data["humidity"]),
                    float(data["moisture"]),
                    soil_encoded,
                    float(data["nitrogen"]),
                    float(data["potassium"]),
                    float(data["phosphorous"]),
                ]
            ],
            columns=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Soil_Type_Encoded",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
            ],
        )

        input_scaled = crop_type_scaler.transform(input_data)
        prediction = crop_type_model_obj.predict(input_scaled)[0]
        probabilities = crop_type_model_obj.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        confidence_str = f"{confidence:.2f}"

        # Save to database
        database.save_prediction("Crop Type Prediction", data, prediction, confidence_str)

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "confidence": confidence_str,
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": "Invalid data type. Please provide numeric values."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
