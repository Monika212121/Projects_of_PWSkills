# Aim - Cement Strength Prediction

4th Advanced Modular structure Project



### Steps to run the project.

## 1.) Setup and Environment 

1. First open Anaconda prompt, Navigate to this empty project folder and then write 'code .' to open this project in VSCode.

2. Now create a new environment by command ```conda create -p {env_name}```. The environment folder will be created in left side e.g ```venv```.

3. Start coding and complete the project. After completing the whole codebase, follow below steps.


## 2.) Installing all the dependencies before running the project.

1. To load dependencies using both requirements.txt and setup.py, here's how you can do it:

```pip install -r requirements.txt```

2. Create the setup.py file and then run the command

```pip install -e .```


## 3.) Runing project and checking result in Web server.

4. First run the command 'python src/pipelines/training_pipeline.py'. It will train the model and create pickle fileS for preprocessor and the best ML model in the artifacts folder.

5. Run this command ```python application.py``` to run the whole application.

6. Go to a Web browser and type ```http://127.0.0.1:5000/predict```. 

7. In this URL, provide the I/Ps to get O/Ps(predictions) and we will know the strength of cement.