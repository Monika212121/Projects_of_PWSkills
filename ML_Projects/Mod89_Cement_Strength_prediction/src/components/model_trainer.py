import os, sys
from dataclasses import dataclass
from sklearn.metrics import r2_score

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    
    def __init__(self):
        self.trained_model_file_path = ModelTrainerConfig()
    

    def evaluate_model(self, X_train, y_train , X_test, y_test, models):
        try:
            logging.info("evaluate_model: STARTS")

            score_list = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                model_name = list(models.keys())[i]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                score_list[model_name] = score


            logging.info("evaluate_model: ENDS")
            return score_list


        except Exception as e:
            logging.info("Error occurred in evaluate_model()")
            raise CustomException(e, sys)



    def initiate_model_training(self, train_arr, test_arr):

        try:
            logging.info("initiate_model_training(): STARTS")

            # Splitting datasets into independent and dependent features
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'SVR': SVR(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor':  GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
            }

            score_list:dict = self.evaluate_model(X_train, y_train, X_test, y_test, models)

            # To find the best model
            max_score = max(score_list.values())

            model_names = list(score_list.keys())
            model_scores = list(score_list.values())

            # To find index of the max score
            index = model_scores.index(max_score)

            # To find the best model (model with max score)
            best_model_name = model_names[index]
            best_model = models[best_model_name]

            logging.info(f"Best accuracy : {max_score}, Best model name : {best_model_name}, Best model : {best_model}")

            save_object(
                file_path= self.trained_model_file_path.trained_model_file_path,
                obj = best_model
            )

            logging.info("initiate_model_training(): ENDS")

  
        except Exception as e:
            logging.info("Error occurred in initiate_model_training()")
            raise CustomException(e, sys)

