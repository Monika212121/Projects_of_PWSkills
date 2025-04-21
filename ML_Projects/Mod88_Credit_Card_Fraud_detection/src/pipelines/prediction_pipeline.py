import os, sys
import pandas as pd

from src.logger import logging
from src.utils import load_object
from src.exception import CustomException


class PredictPipeline:
    
    def __init__(self):
        pass

    def predict_fraud(self, features):
        try:
            logging.info("predict_fraud(): STARTS")

            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            best_model = load_object(model_path)

            # Scaling the new input data
            data_scaled = preprocessor.transform(features)

            # Predicting the O/P for the scaled input data

            y_proba = best_model.predict_proba(data_scaled)
            y_pred = (y_proba[:, 1] > 0.3).astype(int)

            logging.info(f"y_proba of the best model'{best_model}': {y_proba}")
            logging.info(f"RESULT------>>>>{y_pred}")

            return y_pred

        except Exception as e:
            logging.info("Error occurred in predict_fraud()")
            raise CustomException(e, sys)


class CustomData:
    
    def __init__(self,
                 LIMIT_BAL:int,
                 SEX:int,
                 EDUCATION:int,
                 MARRIAGE:int,
                 AGE:int,
                 PAY_0:int,
                 PAY_2:int,
                 PAY_3:int,
                 PAY_4:int,
                 PAY_5:int,
                 PAY_6:int,
                 BILL_AMT1:int,
                 BILL_AMT2:int,
                 BILL_AMT3:int,
                 BILL_AMT4:int,
                 BILL_AMT5:int,
                 BILL_AMT6:int,
                 PAY_AMT1:int,
                 PAY_AMT2:int,
                 PAY_AMT3:int,
                 PAY_AMT4:int,
                 PAY_AMT5:int,
                 PAY_AMT6:int,
                 ):
        
        self.limit_bal = LIMIT_BAL
        self.sex = SEX
        self.education = EDUCATION
        self.marriage = MARRIAGE
        self.age = AGE
        self.pay0 = PAY_0
        self.pay2 = PAY_2
        self.pay3 = PAY_3
        self.pay4 = PAY_4
        self.pay5 = PAY_5
        self.pay6 = PAY_6
        self.bill_amt1 = BILL_AMT1
        self.bill_amt2 = BILL_AMT2
        self.bill_amt3 = BILL_AMT3
        self.bill_amt4 = BILL_AMT4
        self.bill_amt5 = BILL_AMT5
        self.bill_amt6 = BILL_AMT6
        self.pay_amt1 = PAY_AMT1
        self.pay_amt2 = PAY_AMT2
        self.pay_amt3 = PAY_AMT3
        self.pay_amt4 = PAY_AMT4
        self.pay_amt5 = PAY_AMT5
        self.pay_amt6 = PAY_AMT6


    def get_data_as_dataframe(self):
        try:
            
            new_input_data_dict = {
                'LIMIT_BAL': [self.limit_bal],
                'SEX': [self.sex],
                'EDUCATION': [self.education],
                'MARRIAGE': [self.marriage],
                'AGE': [self.age],
                'PAY_0': [self.pay0],
                'PAY_2': [self.pay2],
                'PAY_3': [self.pay3],
                'PAY_4': [self.pay4],
                'PAY_5': [self.pay5],
                'PAY_6': [self.pay6],
                'BILL_AMT1': [self.bill_amt1],
                'BILL_AMT2': [self.bill_amt2],
                'BILL_AMT3': [self.bill_amt3],
                'BILL_AMT4': [self.bill_amt4],
                'BILL_AMT5': [self.bill_amt5],
                'BILL_AMT6': [self.bill_amt6],
                'PAY_AMT1': [self.pay_amt1],
                'PAY_AMT2': [self.pay_amt2],
                'PAY_AMT3': [self.pay_amt3],
                'PAY_AMT4': [self.pay_amt4],
                'PAY_AMT5': [self.pay_amt5],
                'PAY_AMT6': [self.pay_amt6],
            }

            df = pd.DataFrame(new_input_data_dict)
            logging.info(f"New input dataframe shape: {df.shape}")
            return df


        except Exception as e:
            logging.info("Error occurred in get_data_as_df()")
            raise CustomException(e, sys)