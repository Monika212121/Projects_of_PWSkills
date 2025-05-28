import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor_obj(self, df):

        try:
            logging.info("get_preprocessor_obj: STARTS")

            # Separating numerical and categorical features from the dataset
            num_cols = df.select_dtypes(include = ['int64', 'float64']).columns.to_list()
            cat_cols = df.select_dtypes(include = ['object', 'category']).columns.to_list()

            # Creating numerical and categorical Pipelines

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy= 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy= 'moset_frequent')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor =  ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])

            logging.info("get_preprocessor_obj: ENDS")
            return preprocessor
        

        except Exception as e:
            logging.info("Error occurred in get_preprocessor_obj()")
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_df_path, test_df_path):

        try:
            logging.info("initiate_data_transformation: STARTS")

            train_df = pd.read_csv(train_df_path)
            test_df = pd.read_csv(test_df_path)

            # Separating datasets into independent and dependent features
            X_train = train_df.drop(columns = ["Concrete compressive strength(MPa, megapascals) "], axis = 1)
            y_train = train_df["Concrete compressive strength(MPa, megapascals) "]
            X_test = test_df.drop(columns = ["Concrete compressive strength(MPa, megapascals) "])
            y_test = test_df["Concrete compressive strength(MPa, megapascals) "]

            preprocessor_obj = self.get_preprocessor_obj(X_train)

            # Scale the independent features
            X_train_sc = preprocessor_obj.fit_transform(X_train)
            X_test_sc = preprocessor_obj.transform(X_test)

            # Concatanate X and y
            train_arr = np.c_[X_train_sc, np.array(y_train)]            # X_train_sc + y_train
            test_arr = np.c_[X_test_sc, np.array(y_test)]               # X_test_sc + y_test

            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )

            logging.info("initiate_data_transformation: ENDS")

            return(
                train_arr,
                test_arr
            )


        except Exception as e:
            logging.info("Error occurred in initiate_data_transformation()")
            raise CustomException(e, sys)
        