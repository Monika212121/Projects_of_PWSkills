import sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    expected_accuracy = 0.60
    model_config_file_path = os.path.join('config','model_config.yaml')



class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()


    def evaluate_models(self, X_train, X_test, y_train, y_test, models):
        try:

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_test_pred)
                report[list(models.keys())[i]] = score

            return report
        

        except Exception as e:
            logging.info("Error occurred in evaluate_model()")
            raise CustomException(e, sys)
        


    def finetune_best_model(self, best_model_obj:object, best_model_name, X_train, y_train) -> object:
        try:
            conf_file_path = self.model_trainer_config.model_config_file_path
            parameters = self.utils.read_yaml_file(conf_file_path)['model_selection']['model'][best_model_name]['search_param_grid']
            
            grid_search = GridSearchCV(best_model_obj, param_grid = parameters, cv = 5, n_job = -1, verbose = 2)

            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            logging.info(f"Best params from Hyperparameter tuning are: {best_params}")

            # Setting the model with the best hyperparameters, to get maximum accuracy
            finetuned_model = best_model_obj.set_params(**best_params)

            return finetuned_model
            

        except Exception as e:
            logging.info("Error occurred in finetune_best_model()")
            raise CustomException(e,sys)



    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("initiate_model_training(): STARTS")

            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],     # leaving the last target column
                test_array[:,:-1],
                train_array[:,-1],      # taking only last target column
                test_array[:,-1]
            )

            ml_models = {
                'SVC': SVC(),
                'XGBClassifier': XGBClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier()
            }

            # Evaluating all the models' score and finding the best model score and the best model
            model_report = self.evaluate_models(X_train, X_test, y_train, y_test, ml_models)

            best_model_score = max(sorted(model_report.values()))

            model_names = list(model_report.keys())
            model_rep_vals = list(model_report.values())

            best_model_name = model_names[model_rep_vals.index(best_model_score)]

            best_model = ml_models[best_model_name]
            logging.info(f"Before Finetuning: Best model name: {best_model_name}, Best model score: {best_model_score}")

            # Hyperparameter tuning the best model, to get model with the parameters to get maximum accuracy
            best_model = self.finetune_best_model(best_model, best_model_name, X_train, y_train)

            # Finding accuracy for the best finetuned model
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_test_pred)

            logging.info(f"After Finetuning: Best model name: {best_model_name}, Best model score: {best_model_score}")

            # Comparing the best model's accuracy with the expected accuracy 
            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found with an accuracy greater than the provided threshold of 0.60")
            
            # Creating the path for the best trained model and saving it in the created path
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok= True)
            self.utils.save_object(file_path = self.model_trainer_config.trained_model_path, obj = best_model)

            logging.info("initiate_model_training(): ENDS")
            return self.model_trainer_config.trained_model_path             # Not using this path for now, from here


        except Exception as e:
            logging.info("Error occurred in initiate_model_training()")
            raise CustomException(e,sys)