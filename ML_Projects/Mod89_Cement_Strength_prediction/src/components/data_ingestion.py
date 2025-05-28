import os, sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        try:
            logging.info("initiate_data_ingestion: STARTS")
            
            df = pd.read_csv(os.path.join('notebooks', 'data/cement_data.csv'))

            train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

            # Creating the artifacts folder
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok = True)

            # Saving raw, train and test dataframe separately in 'artifacts' folder
            df.to_csv(self.data_ingestion_config.raw_data_path, index = False)
            train_df.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            test_df.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)

            logging.info("initiate_data_ingestion: ENDS")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        

        except Exception as e:
            logging.info("Error occurred in initiate_data_ingestion()")
            raise CustomException(e, sys)
