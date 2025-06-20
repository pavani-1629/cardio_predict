from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define label mapping
label_mapping = {
    0: "No Disease",
    1: "Angina",
    2: "Arrhythmia",
    3: "Heart Failure",
    4: "Myocardial Infarction",
    5: "General Heart Disease"
}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get form data
            input_data = {
                'age': int(request.form['age']),
                'sex': int(request.form['sex']),
                'cp': int(request.form['cp']),
                'trestbps': float(request.form['trestbps']),
                'chol': float(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalach': float(request.form['thalach']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': int(request.form['thal'])
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Predict
            pred = model.predict(input_df)[0]
            prediction = label_mapping.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error occurred: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
