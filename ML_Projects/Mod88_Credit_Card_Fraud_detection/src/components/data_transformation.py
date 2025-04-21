import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Packages for Feature Engineering (FE)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Packages for creating pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Custom packages created in the same project
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


# NOTE: 
# Here our aim is to create a Preprocessor object in which the pre-defined FE is already done.
# We will save (pickle) this Preprocessor object, which will be ready to be applied to any dataset.

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor_object(self, df):
        
        try:
            logging.info("get_preprocessor_object(): STARTS")

            # Separating numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            logging.info(f'\n num cols: {numerical_cols}, cat cols: {categorical_cols}')

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info("get_preprocessor_object(): ENDS")
            return preprocessor


        except Exception as e:
            logging.info("Error occured in get_preprocessor_object()")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_df_path, test_df_path):

        try:
            logging.info("initiate_data_transformation(): STARTS")

            train_df = pd.read_csv(train_df_path)
            test_df = pd.read_csv(test_df_path)

            logging.info(f"train dataframe shape: {train_df.shape}")
            logging.info(f"test dataframe shape: {test_df.shape}")

            # Separating independent and dependent features
            X_train = train_df.drop(columns=['default payment next month'], axis = 1)
            y_train = train_df['default payment next month']

            X_test = test_df.drop(columns=['default payment next month'], axis = 1)
            y_test = test_df['default payment next month']

            preprocessor_obj = self.get_preprocessor_object(X_train)

            # Scaling the train and test data using Preprocessor obj
            X_train_scaled = preprocessor_obj.fit_transform(X_train)
            X_test_scaled = preprocessor_obj.transform(X_test)

            # Concatanating train and test datasets
            train_arr = np.c_[X_train_scaled, np.array(y_train)]        # X_train_scaled + y_train
            test_arr = np.c_[X_test_scaled, np.array(y_test)]           # X_test_scaled + y_test

            # Saving the preprocessor object in the artifacts folder 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("initiate_data_transformation(): ENDS")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error occured in initiate_data_transformation()")
            raise CustomException(e, sys)

