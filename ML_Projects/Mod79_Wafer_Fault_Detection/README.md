# Aim - Wafer Fault Prediction

Second Advanced Modular structure Project


# Steps to follow:

1.) Create an environment for this particular project.(Open Anaconda Prompt->Navigate to this project folder-> Type 'code .' command)

2.) Vscode will be open. Open a cmd terminal and type "conda create -p venv python=3.12.9". Then activate the environment by using "conda activate "project location".

3.) Load all the requirements typing "pip install -r requirements.txt". Mention all the packages in the 'requirements.txt' file.



### Important points about the project:

a.) Run 'python app.py' and go to http://127.0.0.1:5000/predict to check the API fot predicting Diamond price by giving the input parameters(data).

b.) All the main ML code is written inside 'src' folder.

c.) If we want to check if the ML flow is executed well, run the command 'python src/pipelines/training_pipeline.py'.

This will execute all the 3 major processes : Data Ingestion, Data Transforamtion and Model Training.



# 📄✏ Sensor Fault Detection Project
**Brief:** In electronics, a **wafer** (also called a slice or substrate) is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate(serves as foundation for contruction of other components) for microelectronic devices built in and upon the wafer. 

It undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.

#### Dataset is taken from Kaggle and stored in mongodb


💿 Installing
1. Environment setup.
```
conda create --prefix venv python==3.8 -y
```
```
conda activate venv/
````
2. Install Requirements and setup
```
pip install -r requirements.txt
```
5. Run Application
```
python app.py
```

🔧 Built with
- flask
- Python 3.8
- Machine learning
- Scikit learn
- 🏦 Industrial Use Cases

