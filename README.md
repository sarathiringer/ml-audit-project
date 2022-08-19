# Machine Learning Audit Project

Repository for the audit project at Pink Programming Ethical &amp; Responsible ML Summer Camp 2022.

## Goal

The goal is to scrutinize the transparency, fairness, sustainability and data privacy of the model.

## Setup
* 3 groups, one for each topic/theme
* Choose yourself which group to join!
* Evaluate the model in any way you find suitable
* If you have time, make suggestions for improvements
* Present to everyone at 12:00!

## Structure of this directory
The whole project can be run with `salary_audit_project.py`. Data and model is stored under their respective directories. To inspect the preprocessing of the data, check `preprocess.py`. To inspect the model training, check `train.py`.

## Tips
* To load the model artifact, run:
```
from tensorflow import keras
model = keras.models.load_model('model/salary_model')
```

* To load the data files from the `data` directory, run:
```
raw = pd.read_csv('data/adult.csv')
preprocessed = pd.read_csv('data/adult_preprocessed.csv')
```
