import os, sys
import pandas as pd
import pickle
import yaml
#import boto3

from src.exception import CustomException
from src.logger import logging


class MainUtils:

    def __init__(self) -> None:
        pass


    def read_yaml_file(self, filename:str) -> dict:
        try:
            with open(filename, 'rb') as yaml_file:
                return yaml.safe_load(yaml_file)
        
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def read_schema_config_file(self) -> dict:
        try:
            schema_cofig = self.read_yaml_file(os.path.join('config','schema.yaml'))
            return schema_cofig
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    @staticmethod
    def save_object(file_path:str, obj:object) -> None:
        logging.info("MainUtils -> save_object(): STARTS")

        try:
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("MainUtils -> save_object(): ENDS")

        except Exception as e:
            raise CustomException(e, sys) from e
        

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("MainUtils-> load_object(): STARTS")

        try:
            with open(file_path, 'rb') as file_obj:
                obj = pickle.load(file_obj)

            logging.info("MainUtils-> load_object(): ENDS")
            return obj

        except Exception as e:
            logging.info("Error occured in MainUtils-> load_object()")
            raise CustomException(e, sys)
        