import sys, os
import numpy as np
import pandas as pd
from zipfile import Path
from dataclasses import dataclass
#from pymongo import MongoClient

from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
from src.constant import *


@dataclass
class DataIngestionConfig:
    artifacts_folder: str = os.path.join('artifacts')


class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()


    def get_data_from_mongo(self, db_name, collection_name):
        """
        Descr: This helper method retrieves data from mongodb and return this data as dataframe.
        Output: Returns dataset as a pd.DataFrame.
        """

        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))                        # Reads data from the provided database and collection name

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({'na': np.nan}, inplace= True)
            return df 

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_ingestion(self) -> Path:
        """
        Descr: This method gets data from mongodb and saves it in the artifacts folder/raw_file_path.
        Output: Returns the 'raw file path' in string format. 
        """

        try:
            logging.info("initiate_data_ingestion(): STARTS")
            artifacts_path = self.data_ingestion_config.artifacts_folder
            os.makedirs(artifacts_path, exist_ok= True)
            
            #sensor_data = self.get_data_from_mongo(db_name= MONGO_DATABASE_NAME, collection_name= MONGO_COLLECTION_NAME)
            sensor_data = pd.read_csv(os.path.join('notebooks/data','wafer_data.csv'))

            raw_file_path = os.path.join(artifacts_path, 'wafer_fault.csv')
            sensor_data.to_csv(raw_file_path, index= False)                    # Writes 'sensor data' to the .csv file 

            logging.info("initiate_data_ingestion(): ENDS")
            return raw_file_path


        except Exception as e:
            logging.info("Error occured in initiate_data_ingestion()")
            raise CustomException(e, sys)
