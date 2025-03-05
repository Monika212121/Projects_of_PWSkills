import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

# Imported to calculate Performance metrics of the ML model
from sklearn.metrics import r2_score


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train, y_train, x_test, y_test, models):

    try:
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]

            # Train model
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = score

        return report

    except Exception as e:
        logging.info("Error occured in evaluate_model()")
        raise CustomException(e,sys)
    

def load_object(file_path):

    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error occured in load_object()")
        raise CustomException(e,sys)