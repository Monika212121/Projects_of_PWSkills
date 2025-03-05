import pandas as pd
import numpy as np
import sys, os
from dataclasses import dataclass

# For feature engineering / data transformation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

# For creating pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imported from other modules in the same project
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Data transformation config class
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data transformation class
class DataTransformation:

    def __init__(self):
        self.data_transforamtion_config = DataTransformationConfig()

    # Same steps as in Data transformation process in the 'training.ipynb' file
    def get_data_transformation_object(self):     

        try:
            logging.info("get_data_transformation_object(): STARTS")

            # Defining columns to be Scaled or Oridnal encoded
            categorical_cols = ['cut', 'color', 'clarity']              # ordinal
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']  # scaled

            # Define the custom ranking for each ordianl variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")

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
                    ('ordinalencoder', OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

        
            logging.info("Pipeline Completed")
            logging.info("get_data_transformation_object(): ENDS")

            return preprocessor


        except Exception as e:
            logging.info("Error occured in get_data_transformation_object()")
            raise CustomException(e,sys)

        
    def initiate_data_transforamtion(self, train_path, test_path):

        try:
            
            logging.info("initiate_data_transformation(): STARTS")

            # Reading train and test data from their respective paths
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed successfully")
            logging.info(f'Train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test dataframe head: \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformation_object()

            # Segregating into independent and dependent features
            target_col_name = 'price'
            drop_columns = [target_col_name, 'id']

            input_feat_train_df = train_df.drop(columns = drop_columns, axis = 1)    # x_train
            target_feat_train_df = train_df[target_col_name]                        # y_train

            input_feat_test_df = test_df.drop(columns= drop_columns, axis = 1)      # x_test
            target_feat_test_df = test_df[target_col_name]                          # y_test

            # Applying the transformation(preprocessor) in the train and test data
            input_feat_train_arr = preprocessing_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr = preprocessing_obj.transform(input_feat_test_df)
            logging.info("Preprocessor object applied on train and test data")

            # Concatenating (x_train + y_train) and (x_test + y_test) dfs
            train_arr = np.c_[input_feat_train_arr, np.array(target_feat_train_df)]
            test_arr = np.c_[input_feat_test_arr, np.array(target_feat_test_df)]

            # Creating pickle file in the utils package/save_object()
            save_object(
                file_path = self.data_transforamtion_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor object's pickle file created and saved in artifacts folder")
            logging.info("initiate_data_transformation(): ENDS")
            
            return (
                train_arr,
                test_arr,
                self.data_transforamtion_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error occured in initiate_data_transformation()")
            raise CustomException(e,sys)

