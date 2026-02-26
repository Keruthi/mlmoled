import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# --- 1. Initialize Flask Application ---
app = Flask(__name__)

# --- 2. Load model and preprocessors ---
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('one_hot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)

    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

except Exception as e:
    print("Model loading failed:", e)


# --- 3. Home route (to check server) ---
@app.route('/')
def home():
    return "ML API is running successfully"


# --- 4. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Required columns
        categorical_cols = ['Region', 'Country']
        numerical_cols = ['AirQuality']

        # Validate input
        required_cols = categorical_cols + numerical_cols
        for col in required_cols:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct datatype
        input_df[numerical_cols] = input_df[numerical_cols].astype(float)

        # Separate features
        input_categorical = input_df[categorical_cols]
        input_numerical = input_df[numerical_cols]

        # One-hot encoding
        X_cat = one_hot_encoder.transform(input_categorical)
        X_cat_df = pd.DataFrame(
            X_cat,
            columns=one_hot_encoder.get_feature_names_out(categorical_cols)
        )

        # Scaling
        X_num = scaler.transform(input_numerical)
        X_num_df = pd.DataFrame(X_num, columns=numerical_cols)

        # Combine features
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
