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
- link :-webapp(https://diabetes-prediction-using-svm.onrender.com)

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

<img width="1328" height="980" alt="Screenshot 2025-07-15 234703" src="https://github.com/user-attachments/assets/c7f4f251-e08e-445a-8554-d449b35c3229" />

<img width="1308" height="876" alt="image" src="https://github.com/user-attachments/assets/01a20fe7-596b-44c8-b466-71bfe0531fcb" />



