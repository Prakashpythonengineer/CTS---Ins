from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pickle
import pandas as pd
import xgboost as xgb
from flask_socketio import SocketIO, emit
from transformers import pipeline
import torch

app = Flask(__name__)
app.secret_key = "InsuranceGuard"
socketio = SocketIO(app, cors_allowed_origins="*")
@socketio.on('message')
def handle_message(data):
    user_input = data['message']
    messages = [
    {"role": "user", "content": user_input},]
    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium",max_length=256)
    res = pipe(user_input)
    emit('response', {'message': user_input, 'generated_text': res[0]})


with open('Random_Forest_Finalized.pkl', 'rb') as file:
        model1 = pickle.load(file)

with open('Logistics_Regression_Finalized.pkl', 'rb') as file:
    model2 = pickle.load(file)

with open('RG_Boost_Finalized.pkl', 'rb') as file:
    model3 = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/find', methods=['POST'])
def fraud():
    provider_id = int(request.form.get('provider_id'))
    print(provider_id)
    try:
        data = pd.read_csv('finalized_ds.csv')
        print("dataset loaded")
        X=data.drop(columns=['Provider','PotentialFraud'],axis=1)
        y=data['PotentialFraud']
        print("columns loaded")
        i = provider_id
        X = X[i:i+1]
        y = y[i:i+1]
        res = {}
        y_pred1 = model1.predict_proba(X)[0][1]*100
        res['Decision Tree'] = y_pred1
        print("The Percentage of being Fraud is", y_pred1)
        print("The Truth is",y)

        y_pred2 = model2.predict_proba(X)[0][1]*100
        res['Logistic Regression'] = y_pred2
        print("The Percentage of being Fraud is", y_pred2)
        print("The Truth is",y)

        drow = xgb.DMatrix(X)
        y_pred3 = model3.predict(drow)[0]*100
        res['XGBoost'] = y_pred3
        print("The Percentage of being Fraud is",y_pred3)
        
        return render_template('result.html', results=res)
    except Exception as e:
        print("Error")
        return redirect(url_for('home'))


if __name__ == '__main__':
    socketio.run(app, debug=True)
