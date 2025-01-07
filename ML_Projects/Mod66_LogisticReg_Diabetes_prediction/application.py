from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# Import Logistic regressor model and standard scaler via pickle.
standard_scaler = pickle.load( open('models/logistic_scaler.pkl', 'rb') )
logistic_regressor_model = pickle.load( open('models/logisticRegressor.pkl', 'rb') )


# route for homepage.
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdisease', methods = ['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data_scaled = standard_scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        res = logistic_regressor_model.predict(new_data_scaled)

        return render_template('home.html', result = res[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")