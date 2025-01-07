from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# Import Ridge regressor model and standard scaler via pickle
standard_scaler_model = pickle.load( open('models/scaler.pkl', 'rb') )
ridge_model = pickle.load( open('models/ridge.pkl', 'rb') )


# route for homepage.
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('WS'))
        Rain = float(request.form.get('RAIN'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('CLASSES'))
        Region = float(request.form.get('REGION'))

        new_data_scaled = standard_scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        res = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result = res[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")