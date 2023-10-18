
import streamlit as st
import joblib
import pandas as pd
import numpy as np

#load the model and pipline
tree =joblib.load('tree.apk')
full_pipline = joblib.load('full_pipline.apk')

#load the data
startups = pd.read_csv('50startups.csv')

#title
st.title('50 Startups Profit Prediction')
st.write('This App predicits the Startups Profit')

#take the input from user
RD_Spend = st.slider('RD_Spend' , float(startups['RD_Spend'].min()) , float(startups['RD_Spend'].max()))
Administration = st.slider('Administration' , float(startups['Administration'].min()) , float(startups['Administration'].max()))
Marketing_Spend = st.slider('Marketing_Spend' , float(startups['Marketing_Spend'].min()) , float(startups['Marketing_Spend'].max()))
State = st.selectbox('State_n' , ('New_York','California','Florida'))

#store the inputs as dictionary
user_inputs ={ 'RD_Spend': RD_Spend ,
               'Administration' : Administration ,
               'Marketing_Spend' : Marketing_Spend ,
               'State_n' : State  }

#transform the data into a dataframe
features = pd.DataFrame(user_inputs , index=[0])

#pipline
features_prepared = full_pipline.transform(features)

# Predictions
Prediction = tree.predict(features_prepared)[0]

#display the prediction
st.subheader('Prediction : ' )
st.markdown(''' # $ {} ''' .format(round(Prediction),2))
