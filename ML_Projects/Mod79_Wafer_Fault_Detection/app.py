import os, sys
from flask import Flask, render_template, request, send_file

from src.logger import logging
from src.exception import CustomException

from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline



app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Wafer fault prediction page"


@app.route("/train")
def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.activate_training_pipeline()

        return "Training is completed for the wafer fault dataset"


    except Exception as e:
        logging.info("Error occurred in Training pipeline")
        raise CustomException(e, sys)
    


@app.route("/predict", methods = ['POST', 'GET'])
def predict_route():
    try:
        if request.method == 'POST':

            prediction_pipeline = PredictionPipeline(request)
            prediction_pipeline_config = prediction_pipeline.activate_prediction_pipeline()

            logging.info("Prediction for the provided data is completed. Downloading prediction file")

            return send_file(prediction_pipeline_config.prediction_file_path, 
                             download_name= prediction_pipeline_config.prediction_filename,
                             as_attachment= True)
        
        else:
            return render_template('upload_file.html')


    except Exception as e:
        logging.info("Error occurred in Prediction pipeline")
        raise CustomException(e, sys)
    


if __name__ == "__main__":
    app.run(host= "0.0.0.0", port= 5000, debug= True)