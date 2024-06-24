# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 07:14:04 2023

@author: ADMIN
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('E:/calorie burnt/trained_model.sav','rb'))

def calorie_estimation(input_data):

    input_data_as_numpy_array = np.asarray(input_data,dtype=float)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)

    return (prediction[0])


def main():
    #giving a title
    st.title('calorie estimation web App')
    
    #getting the input data from the user
    
    #Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp
    
    Gender=st.text_input('Gender')
    Age=st.text_input('Age')
    Height=st.text_input('Height')
    Weight=st.text_input('Weight')
    Duration=st.text_input('Duration')
    Heart_Rate=st.text_input('Heart_Rate')
    Body_Temp=st.text_input('Body Temp')
    
    # code for prediction
    pre = ''
    
    # creating a button for prediction
    
    if st.button('Burnt Calorie Estimation'):
        pre= calorie_estimation([Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp])
        
        
        
        st.success(pre)
        
        
        
if __name__== '__main__':
    main()