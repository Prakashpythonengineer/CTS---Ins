import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

with open('Random_Forest_Finalized.pkl', 'rb') as file:
    model1 = pickle.load(file)

with open('Logistics_Regression_Finalized.pkl', 'rb') as file:
    model2 = pickle.load(file)

with open('RG_Boost_Finalized.pkl', 'rb') as file:
    model3 = pickle.load(file)


data = pd.read_csv('finalized_ds.csv')

X=data.drop(axis=1,columns=['Provider','PotentialFraud'])
y=data['PotentialFraud']

i = 0
X = X[i:i+1]
y = y[i:i+1]

y_pred1 = model1.predict_proba(X)[0][1]*100
print("The Percentage of being Fraud is", y_pred1)
print("The Truth is",y)

y_pred2 = model2.predict_proba(X)[0][1]*100
print("The Percentage of being Fraud is", y_pred2)
print("The Truth is",y)

drow = xgb.DMatrix(X)
y_pred3 = model3.predict(drow)[0]*100
print("The Percentage of being Fraud is",y_pred3)