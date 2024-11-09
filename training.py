import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import *
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Đọc dataset
data = pd.read_csv('./dataset/obesity_prediction_dataset_final.csv')
X = data[['Gender','Age','Height','Weight','BMI','family_history_with_overweight','CH2O','FAF']]
y = data['Obesity']


# Chia dataset thành training set và test set (training 80% test 20%) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Lưu tập test
# test_data = pd.DataFrame(data=np.column_stack([X_test, y_test]), columns=data.columns)
# test_data.to_csv("test_data.csv", index=False)
# test_data.describe()
# Chia tập train set thành training và validation(training 60% validation 20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# NaiveBayes
# modelNB = CategoricalNB()
# modelNB.fit(X_train,y_train)
# y_pred = modelNB.predict(X_val)
# joblib.dump(modelNB,'./model/modelNB.pkl')

# Decision Tree
# modelDT = DecisionTreeClassifier()
# modelDT.fit(X_train,y_train)
# y_pred = modelDT.predict(X_val)
# joblib.dump(modelDT,'./model/modelDT.pkl')

# Random Forest
modelRF = RandomForestClassifier()
modelRF.fit(X_train,y_train)
y_pred = modelRF.predict(X_val)
joblib.dump(modelRF,'./model/modelRF.pkl')

confusion = confusion_matrix(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc = roc_curve(y_val,y_pred)
print("Confusion matrix: ",confusion)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score:{f1}")


