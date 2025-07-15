from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/diabetes_model.sav')
scaler = joblib.load('model/scaler.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        result = "Person is not diabetic ❌" if prediction[0] == 0 else "Person is diabetic ✅"
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
