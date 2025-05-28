import os, sys
import pickle

from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            return pickle.dump(obj, file_obj)
   
   
    except Exception as e:
        logging.info(f"Error occurred while saving object in this filepath: '{file_path}'")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)


    except Exception as e:
        logging.info(f"Error occurred while loading object in this filepath: {file_path}")
        raise CustomException (e, sys)