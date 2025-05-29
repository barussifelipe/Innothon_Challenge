# Importing the AnomalyTransformer class from the Anomaly_Transformer module
import Anomaly_Transformer as at
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset into pandas dataframe
dataset = pd.read_csv('Basic_model/Data/Basic_model_dataset_quarterhouravg_400days.csv').drop(columns= ["Supply_ID", "is_Regular"])

#Train_test split 
X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=42)
# Initialize the AnomalyTransformer

anomaly_transformer = at.model.AnomalyTransformer()

# Fit the model on the training data


