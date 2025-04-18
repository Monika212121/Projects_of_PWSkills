import sys, os

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



class TrainingPipeline:
        
    def activate_training_pipeline(self):
        """Desc: This method is the training pipeline consisting 3 processes: Data Ingestion, Data Transforamtion and Model training."""
         
        try:
            obj = DataIngestion()
            raw_file_path = obj.initiate_data_ingestion()
            logging.info("Data Ingestion process ends successfully", raw_file_path)

            data_transformation = DataTransformation(raw_file_path)      # data transformation object initialization requires 'raw_file_path '
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info("Data Transformation process ends successfully")

            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)
            logging.info("Model training completed successfully, HURAYYYYYYY")

         
        except Exception as e:
              logging.info("Error occured in activate_training_pipeline()")
              raise CustomException(e, sys)
