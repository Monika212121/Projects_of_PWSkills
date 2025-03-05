import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass 

# Importing Machine Learning models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor

# Imported from other modules in the same project
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
# Model Training configuration class
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


# Model Training class 
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info("initiate_model_training(): STARTS")
            logging.info("Splitting independent and dependent data from train_aray and test_array")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],            # Leaving last column
                train_array[:,-1],              # Taking only last column
                test_array[:,:-1],
                test_array[:,-1]
            )

            ml_models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor()
            } 

            model_report:dict = evaluate_model(x_train, y_train, x_test, y_test, ml_models)
            print(model_report)
            print('\n =======================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            model_names = list(model_report.keys())
            model_rep_vals = list(model_report.values())

            best_model_name = model_names[model_rep_vals.index(best_model_score)]
            
            best_model = ml_models[best_model_name]

            print(f'\n Best model found: {best_model_name}, R2_score: {best_model_score}')
            print('\n =======================================================\n')
            logging.info(f'Best model found: {best_model_name}, R2_score: {best_model_score}')        # printing under braces

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info("initiate_model_training(): ENDS")

        except Exception as e:
            logging.info("Error occured in initiate_model_training()")
            raise CustomException(e,sys)