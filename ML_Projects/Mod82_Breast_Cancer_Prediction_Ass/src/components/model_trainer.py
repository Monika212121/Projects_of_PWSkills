import os, sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

 
@dataclass
class ModelTrainerConfig:
    model_trainer_config_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            score_list = {}

            for i in range(len(models)):

                model = list(models.values())[i]
                model_name = list(models.keys())[i]

                # Training this model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_score = accuracy_score(y_test, y_pred)
                
                score_list[model_name] = model_score

            return score_list


        except Exception as e:
            logging.info("Error occurred in evaluate_models()")
            raise CustomException(e, sys)


    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("inititate_model_training: STARTS")

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            ml_models = {
                'LogisticRegression': LogisticRegression(),
                'RidgeClassifier': RidgeClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClasssifer': RandomForestClassifier(),
                'GradientBoostClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GaussianNB()': GaussianNB(),
            }

            model_report_list: dict = self.evaluate_models(X_train, y_train, X_test, y_test, ml_models)
         
            print("Model report list: ", model_report_list)
            print('\n======================================================================================\n')
            logging.info(f"Model report: {model_report_list}")

            # To get the best model score from the model report list
            best_model_score = max(model_report_list.values())
            
            model_names = list(model_report_list.keys())
            model_scores = list(model_report_list.values())

            # index at which maximum model accuracy score is present
            index = model_scores.index(best_model_score)

            # To get best model
            best_model_name = model_names[index]
            best_model = ml_models[best_model_name]

            logging.info(f'Best model found: {best_model_name}, best model score: {best_model_score}')

            save_object(
                file_path = self.model_trainer_config.model_trainer_config_file_path,
                obj = best_model
            )

            logging.info("inititate_model_training: ENDS")


        except Exception as e:
            logging.info("Error occurred in initiate_model_training()")
            raise CustomException(e, sys)