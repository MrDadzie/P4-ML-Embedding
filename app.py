import pandas as pd
import streamlit as st
import pickle
from datetime import datetime
from datetime import date
import os

#Useful functions


def load_objects(fp):
    'Load ML components and preprocessing object from a pickle file'
    with open(fp, 'rb') as f:
        ml_components = pickle.load(f)
    return ml_components


def getDateFeatures(df,date):
    df['year'] = df['date'].dt.year
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
family = st.text_input('Enter the product family')
store_type = st.text_input('Enter the store type')
date = st.date_input ('Enter date for predictions')



if st.button('Predict'):
    # Dataframe
    df =pd.DataFrame(
    {'date': [date],'store_nbr': [store_nbr],'family':[family], 'store_type':[store_type]

    }
)
    # Work with the date
    df['date'] = df['date'].astype('datetime64')
    df = getDateFeatures(df,'date')
    df = df.drop('date',axis =1)

    #Processing the data
    # num_cols = ['store_nbr','year','month','is_month_start','is_month_end','day','is_weekend','day_of_the_year','quarter','is_quarter_start','is_quarter_end']
    
   


    #setup
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_fp = os.path.join(DIRPATH,  'streamlit_toolkit')
   
    # Execution
    ml_components = load_objects(ml_core_fp)
    
    
    encoder = ml_components['Encoder']
    scaler = ml_components['Scaler']
    model = ml_components['Model']
    df = encoder.transform(df)
    df = scaler.transform(df)

    
    print(f'[Info] InputData in the DataFrame:\n {df.to_markdown()}')
    
    

    #Output predictions
   
    output = model.predict(df)