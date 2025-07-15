# diabetes prediction using svm...
# model  will analyze the  diffrent features of human report data and will  predict that is patient is diabetic or not.

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv(r"C:\Users\acer\OneDrive\Desktop\ML-PROJECTS\DIabetes prediction using SVM\diabetes.csv")

# print(diabetes_dataset.head)
# print(diabetes_dataset.shape)
# print(diabetes_dataset.describe())

# print(diabetes_dataset['Outcome'].value_counts())
# # 0--> non diabetic person  ||  1-->diabetic person

# print(diabetes_dataset.groupby('Outcome').mean())


# seprating features and  target variable
x = diabetes_dataset.drop(columns='Outcome',axis=1)
y = diabetes_dataset['Outcome']


# print(x)
# print(y)

# standardizing features i.e X
scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)
# print(x_standardized)

# spliting in traing and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# traing the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

# evaluating model
x_train_prediction= classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
# print("training data  acccuracy is  :- ", training_data_accuracy)


x_test_predictioin = classifier.predict(x_test)
x_test_accuracy = accuracy_score(x_test_predictioin, y_test)
# print("testing data accuracy is :- ",x_test_accuracy)

input_data=(4,136,70,0,0,31.2,1.182,22)
input_data_numpy_array = np.asarray(input_data)

# reshaping the array as predicting for the  single instance
input_data_reshaped = input_data_numpy_array.reshape(1,-1)

# standardizing the input
std_input = scaler.transform(input_data_reshaped)
# print(std_input)

prediction = classifier.predict(std_input)
print(prediction)
print("person is not diabetic") if prediction == 0 else print("person is diabetic")

joblib.dump(classifier, 'diabetes_model.sav')
joblib.dump(scaler, 'scaler.sav')