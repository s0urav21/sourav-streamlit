import pandas as pd 
import numpy as np 

import pickle

import streamlit as st 

st.title('Hospital Stay Prediction')
st.write('Predict if the patient will stay based on user input')


data=pd.read_csv('validation_data.csv')
st.write(data)
data=data.drop(['Unnamed: 0'], axis=1)
data=data.drop(columns=['case_id','patientid'])


pred_data=pd.DataFrame(columns=data.columns)
pred_data.loc[0,'Age']='11-20'



for i in pred_data.columns:
	if pred_data[i].dtypes=='object':
		pred_data[i]='Nan'

	else:
		pred_data[i]=0

user_input = st.number_input("Hospital_code")
pred_data.loc[0,'Hospital_code']=user_input


user_input=st.text_input('Age')
pred_data.loc[0,'Age']=user_input




model=pickle.load(open('catboost1.pkl', 'rb'))
if st.button('Submit'):
	pred_data['pred']=model.predict(pred_data)
	pred=pred_data.loc[0,'pred']
	st.write(pred)
