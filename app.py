import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('ohe_geo.pkl','rb') as file:
    ohe_geo=pickle.load(file)

with open('sc.pkl','rb') as file:
    sc=pickle.load(file)

st.title("Customer Churn Prediction")

geography=st.selectbox("Geography", ohe_geo.categories_[0].tolist())
gender=st.selectbox("Gender",label_encoder_gender.classes_.tolist())
age=st.slider("Age",18,100)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",["Yes",'No'])
is_active_member=st.selectbox("Is Active Member",["Yes",'No'])

if st.button("Predict Churn"):
    geo_encoded=ohe_geo.transform([[geography]]).toarray()

    gender_encoded=label_encoder_gender.transform([gender])[0]


    numeric_features=np.array([[
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        1 if has_cr_card=='Yes'else 0,
        1 if is_active_member=='Yes'else 0,
        estimated_salary
    ]])

    final_features=np.concatenate([numeric_features,geo_encoded],axis=1)

    final_features_scaled=sc.transform(final_features)

    prediction=model.predict(final_features_scaled)[0][0]

    if prediction<0.5:
        st.write("The customer is not likely to churn.")
    else:
        st.write("The customer is likely to churn.")