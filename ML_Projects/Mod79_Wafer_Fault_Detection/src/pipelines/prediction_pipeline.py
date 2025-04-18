import pickle
import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.constant import *
from src.logger import logging
from src.utils import MainUtils
from src.exception import CustomException

from flask import request


@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname:str = "predictions"
    prediction_filename:str = "predicted_file.csv"
    prediction_file_path = os.path.join(prediction_output_dirname, prediction_filename)

    model_file_path:str = os.path.join('artifacts', 'model.pkl')
    preprocessor_path:str = os.path.join('artifacts', 'preprocessor.pkl')



class PredictionPipeline:
    
    def __init__(self, request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()


    def save_input_files(self) -> str:
        try:
            pred_file_dir = "prediction_artifacts"
            os.makedirs(pred_file_dir, exist_ok= True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)
            return pred_file_path
        

        except Exception as e:
            raise CustomException(e,sys)
        

    def predict(self, Xnew):
        """
        Descr: This helper method transforms the input data using preprocessor and then predict for the input new data.
        Output: Predictions for the new input data
        """
        try:
            preprocessor = self.utils.load_object(self.prediction_pipeline_config.preprocessor_path)
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)

            Xnew_trans = preprocessor.tranform(Xnew)
            predictions = model.predict(Xnew_trans)

            return predictions
            

        except Exception as e:
            raise CustomException(e, sys)


    def get_predicted_df(self, input_csv_path):
        """
        Descr: This helper method returns the dataframe with a new column, containing predictions
        Output: Predicted dataframe
        """
        try:
            prediction_col: str = TARGET_COLUMN
            input_df : pd.DataFrame = pd.read_csv(input_csv_path)
            
            input_df = input_df.drop(columns= 'Unnamed: 0') if 'Unnamed: 0' in input_df.columns else input_df

            predictions = self.predict(input_df)
            input_df[prediction_col] = [pred for pred in predictions]

            # Target column(quality = 0/1) mapping
            target_col_mapping = {0:'bad', 1:'good'}
            input_df[prediction_col] = input_df[prediction_col].map(target_col_mapping)

            # Saving the resulted prediction df in the predictions/predicted_file.csv
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok = True)
            input_df.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)

            logging.info(f"prediction output file is created and saved in: {self.prediction_pipeline_config.prediction_file_path}")
            return


        except Exception as e:
            raise CustomException(e, sys)
        

    def activate_prediction_pipeline(self):
        """
        Descr: The main method for running the predction pipleine
        Output: The prediction df will be created and saved
        """
        try:
            logging.info("run_pipeline(): STARTS")

            input_csv_path = self.save_input_files()
            self.get_predicted_df(input_csv_path)

            logging.info("run_pipeline(): ENDS")

            return self.prediction_pipeline_config
        

        except Exception as e:
            logging.info("Error occurred in activate_prediction_pipeline()")
            raise CustomException(e,sys)