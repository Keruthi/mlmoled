import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# --- 1. Initialize Flask Application ---
app = Flask(__name__)

# --- 2. Load model and preprocessors ---
model = None
scaler = None
one_hot_encoder = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('one_hot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)

    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully ✅")

except Exception as e:
    print("Model loading failed:", e)


# --- 3. Home route ---
@app.route('/')
def home():
    return "ML API is running successfully 🚀"


# --- 4. Prediction Endpoint ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        categorical_cols = ['Region', 'Country']
        numerical_cols = ['AirQuality']

        # ✅ If GET request → show sample prediction
        if request.method == 'GET':
            data = {
                "Region": "Asia",
                "Country": "India",
                "AirQuality": 80
            }
        else:
            data = request.get_json()

        # Validate input
        for col in categorical_cols + numerical_cols:
            if col not in data:
                return jsonify({"error": f"Missing field: {col}"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        input_df[numerical_cols] = input_df[numerical_cols].astype(float)

        # Preprocessing
        X_cat = one_hot_encoder.transform(input_df[categorical_cols])
        X_cat_df = pd.DataFrame(
            X_cat,
            columns=one_hot_encoder.get_feature_names_out(categorical_cols)
        )

        X_num = scaler.transform(input_df[numerical_cols])
        X_num_df = pd.DataFrame(X_num, columns=numerical_cols)

        X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

        # Prediction
        prediction = model.predict(X_processed)
        proba = model.predict_proba(X_processed)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": float(proba[0][0]),
            "probability_class_1": float(proba[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- 5. Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
