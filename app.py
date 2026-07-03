import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder    

model=load_model('model.h5')
with open('label_encode_gender.pkl','rb') as file:
    label_gender = pickle.load(file)
with open('label_onehot_encoder_geo.pkl','rb') as file:
    label_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file) 

##streamlit app
st.title("Bank Customer Churn Prediction")
st.write("Enter the details of the customer to predict whether they will leave the bank or not.")
#input data from user
geography=st.selectbox("Geography",label_geo.categories_[0])
gender=st.selectbox("Gender",label_gender.classes_)
credit_score=st.number_input("Credit Score",min_value=0,max_value=1000)
age=st.number_input("Age",min_value=0,max_value=120)
tenure=st.number_input("Tenure",min_value=0,max_value=10)
balance=st.number_input("Balance",min_value=0.0)
num_of_products=st.slider("Number of Products",1,4)
estimated_salary=st.number_input("Estimated Salary")
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])
#input_data=np.array([[geography,gender,credit_score,age,tenure,balance,num_of_products,has_cr_card,is_active_member]])
#encode the input data  

input_data= pd.DataFrame({
    'Gender': [label_gender.transform([gender])[0]],
    'CreditScore': [credit_score], 
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'EstimatedSalary': [estimated_salary],
    'IsActiveMember': [is_active_member]

})
geo_encoded=label_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
#input data scaling
input_data_scaled=scaler.transform(input_data)

get_pred=model.predict(input_data_scaled)
get_pr=get_pred[0][0]
st.write(f"Predicted Probability of Churn: {get_pr:.2f}")
if get_pr>0.5:
    st.write("The person will leave the bank")
else:
    st.write("The person will stay with the bank")