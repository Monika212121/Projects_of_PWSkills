import os, sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


# Create the data ingestion configuration class
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')


# Create the data ingestion class
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion: STARTS")

        try:
            # Getting data from notebooks folder
            df = pd.read_csv(os.path.join('notebooks/data', 'breast_cancer_dataset.csv'))
            
            # Train test splitting the dataset
            train_df, test_df = train_test_split(df, train_size = 0.20, random_state = 42)

            # Creating the artifacts folder 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)

            # Saving data in the artifacts folder in 3 different files, i.e. raw, train, test respectively
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            train_df.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_df.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion: ENDS")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )            


        except Exception as e:
            logging.info("Error occurred in initiate_data_ingestion()")
            raise CustomException(e, sys)