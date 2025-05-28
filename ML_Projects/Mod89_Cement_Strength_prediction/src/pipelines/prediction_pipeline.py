import os, sys
import pandas as pd

from src.logger import logging
from src.utils import load_object
from src.exception import CustomException 

'''
    def __init__(self, 
                Cement (component 1)(kg in a m^3 mixture): float,
                Blast Furnace Slag (component 2)(kg in a m^3 mixture): float,
                Fly Ash (component 3)(kg in a m^3 mixture): float,
                Water (component 4)(kg in a m^3 mixture): float,
                Superplasticizer (component 5)(kg in a m^3 mixture): float,
                Coarse Aggregate (component 6)(kg in a m^3 mixture): float,
                Fine Aggregate (component 7)(kg in a m^3 mixture): float,
                Age (day): int,
                ):
'''

class CustomData:
    def __init__(self, 
                cement : float,
                blast_furnace_slag : float,
                fly_ash : float,
                water : float,
                superplasticizer : float,
                coarse_aggregate : float,
                fine_aggregate : float,
                age: int,
                ):
        
        self.cement = cement
        self.blast_furnace_slag = blast_furnace_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age = age
        

    def get_data_as_dataframe(self):
        try:
            logging.info("get_data_as_dataframe(): STARTS")

            new_data = {
                'cement' : [self.cement],
                'blast_furnace_slag' : [self.blast_furnace_slag],
                'fly_ash' : [self.fly_ash],
                'water' : [self.water],
                'superplasticizer' : [self.superplasticizer],
                'coarse_aggregate' : [self.coarse_aggregate],
                'fine_aggregate' : [self.fine_aggregate],
                'age' : [self.age]
            }

            features_df = pd.DataFrame(new_data)

            logging.info(f"New data df: {features_df}")
            logging.info("get_data_as_dataframe(): ENDS")

            return features_df


        except Exception as e:
            logging.info("Error occurred in get_data_as_dataframe()")
            raise CustomException(e, sys)



class PredictPipeline:
    
    def __init__(self):
        pass

    def predict_cement_strength(self, features_df):

        try:
            logging.info("predict_cement_strength(): STARTS")

            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            best_model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(file_path = preprocessor_path)
            best_model = load_object(file_path = best_model_path)
            
            logging.info(f"New cement data before transformation: {features_df}")
            
            new_data_scaled = preprocessor.transform(features_df)       # Scaling the new data for prediction
            y_pred = best_model.predict(new_data_scaled)
            
            logging.info(f"New cement data : {features_df}")
            logging.info(f"\nPredict cement strength : {y_pred}")

            logging.info("predict_cement_strength(): ENDS")

            return y_pred


        except Exception as e:
            logging.info("Error occurred in predict_cement_strength()")
            raise CustomException(e, sys)


# Cement (component 1)(kg in a m^3 mixture),Blast Furnace Slag (component 2)(kg in a m^3 mixture),Fly Ash (component 3)(kg in a m^3 mixture),Water  (component 4)(kg in a m^3 mixture),Superplasticizer (component 5)(kg in a m^3 mixture),Coarse Aggregate  (component 6)(kg in a m^3 mixture),Fine Aggregate (component 7)(kg in a m^3 mixture),Age (day),"Concrete compressive strength(MPa, megapascals) "
