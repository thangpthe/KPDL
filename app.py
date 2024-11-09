from flask import Flask,render_template,request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

model = joblib.load("modelRF.pkl")
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bmi = float(request.form['bmi'])
    overweight_history = int(request.form['overweight'])
    ch20 = int(request.form['ch20'])
    faf = int(request.form['faf'])
    data = np.array([[gender,age,height,weight,bmi,overweight_history,ch20,faf]])
    my_prediction = model.predict(data)
    if int(my_prediction) == 1:
        prediction = "Bạn bị béo phì"
    elif int(my_prediction) == 0:
        prediction = "Bạn không bị béo phì"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)