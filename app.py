
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# --- 1. Load Preprocessing Objects and Model ---
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the one-hot encoder
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- 2. Initialize Flask Application ---
app = Flask(__name__)

# --- 3. Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert input data to pandas DataFrame
        # Ensure the order of columns matches the training data
        # and create a DataFrame with a single row for prediction
        input_df = pd.DataFrame([data])

        # Define the original categorical and numerical column names
        # These should match the columns used during training
        categorical_cols_trained = ['Region', 'Country'] # Based on previous notebook steps
        numerical_cols_trained = ['AirQuality']         # Based on previous notebook steps

        # Separate categorical and numerical columns from the input
        input_categorical = input_df[categorical_cols_trained]
        input_numerical = input_df[numerical_cols_trained]

        # Apply One-Hot Encoding to categorical features
        # Use transform, not fit_transform
        X_categorical_encoded_input = one_hot_encoder.transform(input_categorical)
        X_categorical_df_input = pd.DataFrame(X_categorical_encoded_input, 
                                              columns=one_hot_encoder.get_feature_names_out(categorical_cols_trained))

        # Apply StandardScaler to numerical features
        # Use transform, not fit_transform
        X_numerical_scaled_input = scaler.transform(input_numerical)
        X_numerical_df_input = pd.DataFrame(X_numerical_scaled_input, columns=numerical_cols_trained)

        # Combine preprocessed features
        # Ensure columns are in the same order as X_processed used during training
        # Get the columns from the one_hot_encoder.get_feature_names_out()
        # The order of columns for X_processed was numerical_cols then categorical_cols
        processed_cols = numerical_cols_trained + list(one_hot_encoder.get_feature_names_out(categorical_cols_trained))
        
        X_processed_input = pd.concat([X_numerical_df_input, X_categorical_df_input], axis=1)

        # Reindex to ensure the columns are in the exact order as during training
        # (X_processed from training was `X_processed.columns`)
        # Note: In a real scenario, you'd save X_processed.columns during training and load it here
        # For this example, we'll assume the column order created by concat is consistent if feature names are consistent
        # If X_processed.columns was saved, you would do: X_processed_input = X_processed_input.reindex(columns=saved_X_processed_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(X_processed_input)
        prediction_proba = model.predict_proba(X_processed_input)

        # Return prediction as JSON response
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': prediction_proba[0][0],
            'probability_class_1': prediction_proba[0][1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- 4. Run the Flask app ---
if __name__ == '__main__':
    # In a production environment, use a more robust server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
