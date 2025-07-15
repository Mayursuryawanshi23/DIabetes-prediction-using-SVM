Diabetes Prediction Web App

This is a simple, professional web application that predicts whether a person is diabetic or not based on health parameters. It uses a machine learning model (SVM).

---

##Features

- Built with **Python, Flask, HTML, Tailwind CSS**
- Uses an SVM model for classification
- Accepts user input like:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Predicts and displays whether the person is diabetic or not
- Deployed online using [Render](https://render.com)

---

##  Machine Learning Model

- Model: **Support Vector Machine (SVM)** with linear kernel
- Dataset: PIMA Indians Diabetes Dataset
- Preprocessing: Feature scaling using `StandardScaler`
- Model saved with `joblib`

---

## Tech Stack

| Layer     | Tech                    |
|-----------|-------------------------|
| Frontend  | HTML5, Tailwind CSS     |
| Backend   | Flask (Python)          |
| ML Model  | Scikit-learn (SVM)      |
| Deployment| Render                  |

---

##  Folder Structure

diabetes_prediction_app/
├── app.py
├── requirements.txt
├── model/
│ ├── diabetes_model.sav
│ └── scaler.sav
├── templates/
│ ├── index.html
│ └── result.html



