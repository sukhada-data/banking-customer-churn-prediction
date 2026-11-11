import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib

customer = pd.read_csv(r'D:\ML_Datasets\16_Decision_Tree\Banking_Customer_Churn.csv')

customer.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], inplace = True)

categorical_features = ['Geography', 'Gender']

enc = OneHotEncoder(sparse_output = False)
enc.set_output(transform = 'pandas')
encoded_columns = enc.fit_transform(customer[categorical_features])
other_columns = customer.drop(columns = categorical_features)
encoded_customer = pd.concat([other_columns, encoded_columns], axis = 1)
print(encoded_customer.head())

X, y = encoded_customer.drop(columns = 'Exited'), encoded_customer['Exited']

model = DecisionTreeClassifier(max_depth  = 5)

model.fit(X, y)

filepath = r'D:\ML_Datasets\Deployment\1_Streamlit\4_ML_Model\4_banking_customer_churn'
filename = filepath + r'\churn_model.joblib'
artifacts = {'model' : model, 'encoder': enc}
joblib.dump(artifacts, filename)