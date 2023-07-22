#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import streamlit as st
from streamlit_lottie import st_lottie


# In[2]:


import joblib


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


# In[4]:


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# In[5]:


utm = Image.open("Logo/UTM.png")
meditec = Image.open("Logo/MEDiTEC.png")
ihumen = Image.open("Logo/iHumEn.png")
leftmost, middle1, middle2, right, rightmost = st.columns(5)
fyptitle = 'Prediction of Quercetin Bioactivity Towards Specific Cancer Receptors Using Machine Learning'
drug = 'Drug Details'
target = 'Target Details'
active = 'Prediction: This interaction is active'
inactive = 'Prediction: This interaction is inactive'
intermediate = 'Prediction: This interaction is intermediate'
model = 'XGBoost Classifier (XGBC)'
name = 'Adibah binti Zairul Nisha (A19EB0001)'
data = pd.read_csv('Training  Dataset.csv')
gif = 'https://assets2.lottiefiles.com/packages/lf20_qufi1zre.json'

with st.container():
    with leftmost:
        st.image(utm)
    
    with right:
        st.image(ihumen)
    
    with rightmost:
        st.image(meditec)
        
with st.container():
    st.title("Prediction of Quercetin Bioactivity ")
    st.write(f"This webpage was created as a part of a study titled **{fyptitle}** for Final Year Project 2 (FYP2) 2022/2023-2")
    st.write(f"Developed by: **{name}**")

with st.container():
    st.write('---')
    
    st.write(f'This webpage will predict the bioactivity of protein-ligand complex via **{model}** algorithm')
    if st.checkbox('Show Training Dataset'):
        data


#st.header("Quercetin Bioactivity Prediction Webpage")
with st.container():
    st.write("---")
    left, right = st.columns(2)
    
    with left:
        st.write (f"**{target}**")
        V = int(st.number_input("Valine", min_value=0))
        H = int(st.number_input("Histidine", min_value=0))
        K = int(st.number_input("Lysine", min_value=0))
        S = int(st.number_input("Serine",min_value=0))
        M = int(st.number_input("Methionine",min_value=0))
    
    with right:
        st.write(f"**{drug}**")
        NumHDonors = int(st.number_input("Number of Hydrogen Donors",min_value=0))
        NumHeteroatoms = int(st.number_input("Number of Heteroatoms", min_value=0))
        ExactMolWt = float(st.number_input("Molecular Weight",min_value=0))
        NumSaturatedCarbocycles = int(st.number_input("Number of Saturated Carbocycles",min_value=0))
        NumSaturatedRings = int(st.number_input("Number of Saturated Rings", min_value=0))


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    xgbc = joblib.load("xgbc.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[V,H,NumHDonors,NumHeteroatoms,ExactMolWt,K,S,NumSaturatedCarbocycles,M,NumSaturatedRings]], 
                     columns = ["V", "H", "NumHDonors","NumHeteroatoms","ExactMolWt","K",'S',"NumSaturatedCarbocycles","M","NumSaturatedRings"])
    
    # Get prediction
    prediction = xgbc.predict(X)[0]
    
    with st.container():
        
        st.write('---')

        bottomleft, bottomright = st.columns((3,2))

        with bottomleft:
            # Output prediction
            if (prediction == 0):
                st.write(f'<p style="font-family:Courier;font-size:30px;color:green;">{active}</p>', unsafe_allow_html=True)
            
            if (prediction == 1):
                st.write(f'<p style="font-family:Courier;font-size:30px;color:red;">{inactive}</p>', unsafe_allow_html=True)
        
            if (prediction == 2):
                st.write(f'<p style="font-family:Courier;font-size:30px;color:orange;">{active}</p>', unsafe_allow_html=True)
        
        with bottomright:
            st.lottie(gif)
            
with st.container():
    st.write('---')
    st.write(f"**{fyptitle}**")
    st.write(f"Developed by: **{name}** for Final Year Project 2 (FYP2) 2022/2023-2")

