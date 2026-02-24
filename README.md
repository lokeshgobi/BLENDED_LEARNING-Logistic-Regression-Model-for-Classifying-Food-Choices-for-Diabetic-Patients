# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate features and target, scale the features using MinMaxScaler, and encode the target labels using LabelEncoder.
2. Split the dataset into training and testing sets using train_test_split() with stratified sampling.
3. Train a Logistic Regression model with L2 regularization (multinomial) on the training data and make predictions on the test data.
4. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix, then visualize the confusion matrix using a heatmap. 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: LOKESHWARAN G
RegisterNumber: 212225040210 
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('food_items (1).csv')
print("Name: LOKESHWARAN G")
print("Reg. No: 212225040210")
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)
l2_model = LogisticRegression(
    random_state=123,
    penalty='l2',           
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
## Output:

![alt text](<Screenshot 2026-02-24 141034.png>)

## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
