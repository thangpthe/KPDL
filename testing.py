import pandas as pd
import joblib
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc tập test
test_data = pd.read_csv('test_data.csv')

# Đọc model
# Random Forest
model = joblib.load('./model/modelRF.pkl') 

# Decision Tree
# model = joblib.load('./model/modelDT.pkl') 

# Naive Bayes
# model = joblib.load('./model/modelNB.pkl')

# Các thuộc tính
new_X = test_data[['Gender','Age','Height','Weight','BMI','family_history_with_overweight','CH2O','FAF']]

# Nhãn 
true_y = test_data['Obesity']

# Dự đoán trên tập test
new_y_pred = model.predict(new_X)
print(new_y_pred)

accuracy = accuracy_score(true_y, new_y_pred)
precision = precision_score(true_y, new_y_pred)
recall = recall_score(true_y,new_y_pred)
f1 = f1_score(true_y,new_y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Cross validation
# scores = cross_val_score(model, new_X, true_y, cv=5)
# scores = cross_val_score(model, new_X, true_y, cv=10)
# print(f"Cross-validation Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")


# correlation_matrix = new_X.corr()

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# plt.title('Ma trận tương quan')
# plt.show()

# confusion_matrix = confusion_matrix(true_y, new_y_pred)
# sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

from sklearn.metrics import classification_report

# Tính toán các chỉ số đánh giá
# print(classification_report(true_y, new_y_pred))

# Vẽ đường cong ROC
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(true_y, new_y_pred)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()