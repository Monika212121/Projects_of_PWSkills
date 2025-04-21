# Aim - Credit card Fraud detection

3rd Advanced Modular structure Project

### Steps to run the project.

1. First open Anaconda prompt, Navigate to this empty project folder and then write 'code .' to open VSCode for this project.

2. Now create a new environment by conda create -p {env_name}. The environment folder will be created in left side.

3. Start coding and complete the project. After completing the whole codebase, follow below steps.

4. First run the command 'python src/pipelines/training_pipeline.py'. It will train the model and create pickle fileS for preprocessor and the best ML model in the artifacts folder.

5. Run this command 'python application.py'.

6. Go to a Web browser and type 'http://127.0.0.1:5000/predict'. 

7. In this URL, provide the I/Ps to get O/Ps(predictions) and we will know the credit card holder is fraudster or not.