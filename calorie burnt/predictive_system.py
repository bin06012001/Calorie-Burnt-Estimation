# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model

loaded_model=pickle.load(open('E:/calorie burnt/trained_model.sav','rb'))

input_data = ([[1,36,100.0,50.0,23.0,100.0,40.7]])
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)

print(prediction[0])

