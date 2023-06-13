import pandas as pd
import streamlit as st
import pickle
from datetime import datetime
from datetime import date
import os

#Useful functions
def load_ML_components(fp):
    'load ML components to re-use in the app'
    with open(fp, 'rb')as f:
        object = pickle.load(f)
    return object



def getDateFeatures(df,date):
    df['month'] = df['date'].dt.month
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['date'].dt.isocalendar().week
    df['day_of_the_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    return df

#Interface
#Setting headers
st.title(':green[ML Interface]')
st.header(':green[Sales Predictions Forecast]')

#Input fields
store_nbr = st.number_input('Enter the store number')
family = st.text_input('Enter the producr family')
date = st.date_input ('Enter date for predictions')



if st.button('Predict'):
    # Dataframe
    df =pd.DataFrame(
    {'date': [date],'store_nbr': [store_nbr],'family':[family]

    }
)
    # Work with the date
    df['date'] = df['date'].astype('datetime64')
    df = getDateFeatures(df,'date')
    df = df.drop('date',axis =1)

    #setup
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_fp = os.path.join(DIRPATH,'exports','ml.pkl')

    #execution
    model = load_ML_components(fp =ml_core_fp)

    print(f'[Info] InputData in the DataFrame:\n {df.to_markdown()}')
    
    

    #Output predictions
   
    output = model.predict(df)
