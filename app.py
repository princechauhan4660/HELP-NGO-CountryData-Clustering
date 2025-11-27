import numpy as np
import streamlit as st
import pandas as pd
import joblib

# lets load the joblib instances over here
with open('pipeline.joblib','rb') as file :
    preprocess = joblib.load(file)

with open('model.joblib','rb') as file :
    model = joblib.load(file)

    # lets take the inputs from the user
    st.title('HELP NGO Oraganization')
    st.subheader('This application will help to indetify the development category of a country using social economic factors . original data has been clustered using KMeans')

    # lest take the inputs 
    gpp = st.number_input('Enter the GPP of a country (GDP per population)')
    income = st.number_input('Enter income per population')
    imports = st.number_input('Imports of goods and services per capita')
    exports = st.number_input('Exports of goods and services per capita')
    inflation = st.number_input('The measurement of the annual growth rate of the Total GD')
    lf_expcy = st.number_input('The average number of years a new born child would live if the current mortality patterns are to remain the same')
    fert = st.number_input('The number of children that would be born to each woman if the current age-fertility rates remain the same')
    health = st.number_input('Total health spending per capita. Given as %age of GDP per capita')
    chld_mort = st.number_input('Death of children under 5 years of age per 1000 live births')



    input_list = [chld_mort,exports,health,imports,income,inflation,lf_expcy,fert,gpp]

    final_input_list = preprocess.transform([input_list])

    if st.button('Predict'):
        prediction = model.predict(final_input_list)[0]
        if prediction ==0:
            st.success('Developing')
        elif prediction == 1:
            st.success('Developed')
        else:
            st.success('UnderDevelopment')