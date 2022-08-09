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


user_input=st.text_input('Hospital_type_code')
pred_data.loc[0,'Hospital_type_code']=user_input

user_input=st.number_input('City_Code_Hospital')
pred_data.loc[0,'City_Code_Hospital']=user_input

user_input=st.text_input('Hospital_region_code')
pred_data.loc[0,'Hospital_region_code']=user_input

user_input=st.number_input('Available Extra Rooms in Hospital')
pred_data.loc[0,'Available Extra Rooms in Hospital']=user_input


user_input=st.text_input('Department')
pred_data.loc[0,'Department ']=user_input

user_input=st.text_input('Ward_Type')
pred_data.loc[0,'Ward_Type']=user_input

user_input=st.text_input('Ward_Facility_Code')
pred_data.loc[0,'Ward_Facility_Code']=user_input


user_input=st.number_input('Bed Grade')
pred_data.loc[0,'Bed Grade']=user_input

user_input=st.number_input('City_Code_Patient')
pred_data.loc[0,'City_Code_Patient']=user_input

user_input=st.text_input(' Type of Admission')
pred_data.loc[0,' Type of Admission']=user_input

user_input=st.number_input(' Severity of Illness')
pred_data.loc[0,' Severity of Illness']=user_input

user_input=st.number_input(' Visitors with Patient')
pred_data.loc[0,' Visitors with Patient']=user_input

user_input=st.number_input('  Admission_Deposit')
pred_data.loc[0,'  Admission_Deposit']=user_input


model=pickle.load(open('catboost1.pkl', 'rb'))
if st.button('Submit'):
	pred_data['pred']=model.predict(pred_data)
	pred=pred_data.loc[0,'pred']
	st.write(pred)
