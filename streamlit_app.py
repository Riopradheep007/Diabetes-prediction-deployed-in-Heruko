

import numpy as np
import pickle5
import pandas as pd
#from flasgger import Swagger
import streamlit as st 



pickle_in = open("diabetes_prediction.pkl","rb")
classifier=pickle5.load(pickle_in)



def predict(preganencies_No, Glucose,Bloodpressure,Skinthickness, Insulin, Bmi,Diabetes_pedigree_function,Age):

   
    prediction=classifier.predict([[preganencies_No, Glucose,Bloodpressure,Skinthickness, Insulin, Bmi,Diabetes_pedigree_function,Age]])
    print(prediction)
    return prediction



def main():
    st.title("Diabetes Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    preganencies_No = st.text_input("preganencies_No")
    Glucose= st.text_input("Glucose")
    Bloodpressure= st.text_input("Bloodpressure")
    Skinthickness = st.text_input("Skinthickness ")
    Insulin= st.text_input("Insulin")
    Bmi= st.text_input(" Bmi")
    Diabetes_pedigree_function= st.text_input("Diabetes_pedigree_function")
    Age= st.text_input(" Age")
    result=""
    if st.button("Predict"):
        result=predict(preganencies_No, Glucose,Bloodpressure,Skinthickness, Insulin, Bmi,Diabetes_pedigree_function,Age)
        if result==1:
            st.success("You have a diabetes")
        else:
            st.success("You Dont have diabetes")
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        

if __name__=='__main__':
    main()
