import os, sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            return pickle.dump(obj, file_obj)


    except Exception as e:
        logging.info("Error occured in save_object()")
        return CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    

    except Exception as e:
        logging.info("Error occured in load_object()")
        return CustomException(e, sys)
    