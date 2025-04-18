import sys, os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer

from src.constant import *
from src.logger import logging
from src.utils import MainUtils
from dataclasses import dataclass
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    transformed_train_data_path = os.path.join('artifacts', 'train.npy')
    transformed_test_data_path = os.path.join('artifacts', 'test.npy')
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Function to replace 'na' with np.nan
def handle_na_values(X):
    if isinstance(X, pd.DataFrame):  
        return X.replace('na', np.nan)
    elif isinstance(X, np.ndarray):  
        return np.where(X == 'na', np.nan, X)
    return X

# Function to select only numeric columns
def select_numeric_columns(X):
    if isinstance(X, pd.DataFrame):
        return X.select_dtypes(include=[np.number])
    return X  


class DataTransformation:

    def __init__(self, raw_file_path):
        
        self.data_transformation_config = DataTransformationConfig()
        self.raw_file_path = raw_file_path
        self.utils = MainUtils()


    @staticmethod
    def get_data_from_artifacts(raw_file_path:str) -> pd.DataFrame:
        """
        Descr: This method reads the validated raw data from the provided 'raw_file_path' in artifacts folder.
        Output: Returns the pd.DataFrame containing the merged data.
        """
        try:
            df = pd.read_csv(raw_file_path)
            df.rename(columns = {'Good/Bad': TARGET_COLUMN}, inplace = True)

            return df


        except Exception as e:
            logging.info("Error occured in get_data_from_artifacts()")
            raise CustomException(e, sys)
        

    def get_data_transformation_obj2(self):
        """
        Description: Creates a preprocessing pipeline with Imputation and Scaling.
        Output: Returns the preprocessor object.
        """
        try:

            # Creating the preprocessor pipeline
            preprocessor = Pipeline(steps=[
                ('numeric_selector', FunctionTransformer(select_numeric_columns, validate=False)),  # Select numeric columns
                ('nan_replacement', FunctionTransformer(handle_na_values, validate=False)),  # Handle 'na'
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Fill missing values
                ('scaler', RobustScaler())  # Scale features
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Error occurred in get_data_transforamtion_obj()")
            raise CustomException(e, sys)
        

        
    def get_data_transformation_obj(self):
        """
        Descr: This method create the preprocessor object with Imputationa and Standarization.
        Output: Returns the preprocessor object.
        """
        try:

            # handle_na_values = lambda x: np.where(x =='na' or x == '', np.nan, x)
            # ('nan_replacement', FunctionTransformer(handle_na_values, validate=False)),

            preprocessor = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'constant', fill_value = 0)),
                    ('scaler', RobustScaler())
                ]
            )

            return preprocessor
        

        except Exception as e:
            logging.info("Error occurred in get_data_transforamtion_obj()")
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self):

        try:
            logging.info("initiate_data_transformation(): STARTS")

            df = self.get_data_from_artifacts(raw_file_path = self.raw_file_path)

            X = df.drop(columns = TARGET_COLUMN)
            y = df[TARGET_COLUMN]

            y = np.where(df[TARGET_COLUMN]==-1, 0, 1)   # if df['quality']==(-1), then df['quality'] = 0, else 1, For Model training

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42)

            preprocessor = self.get_data_transformation_obj()

            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)
            
            # Concatenating(X_train + y_train) and (X_test + y_test)
            train_arr = np.c_[X_train_trans, np.asarray(y_train)]
            test_arr =  np.c_[X_test_trans, np.asarray(y_test)]

            preprocessor_path = self.data_transformation_config.preprocessor_obj_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok= True)

            self.utils.save_object(file_path= preprocessor_path, obj= preprocessor)

            logging.info("initiate_data_transformation(): ENDS")
            return train_arr, test_arr, preprocessor_path


        except Exception as e:
            logging.info("Error occurred in initiate_data_transformation()")
            raise CustomException(e, sys)