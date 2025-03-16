from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and encoder
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# Feature names used in training
numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
categorical_features = ['Location', 'Condition', 'Garage']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from form
        input_data = {}
        for feature in numerical_features + categorical_features:
            input_data[feature] = request.form[feature]

        # Convert numerical inputs to float
        for feature in numerical_features:
            input_data[feature] = float(input_data[feature])

        # Convert input into DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        input_categorical = pd.DataFrame(encoder.transform(input_df[categorical_features]), 
                                         columns=encoder.get_feature_names_out())
        
        # Scale numerical variables
        input_numerical = pd.DataFrame(scaler.transform(input_df[numerical_features]), 
                                       columns=numerical_features)

        # Combine processed numerical & categorical features
        input_processed = pd.concat([input_numerical, input_categorical], axis=1)

        # Make prediction
        predicted_price = model.predict(input_processed)[0]

        return render_template("index.html", prediction_text=f"Estimated House Price: ${predicted_price:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
