import streamlit as st
import math as math

# app to demo session states.
col1,col2 = st.columns(2)

with col1:
    angle = st.slider("Choose angle (degrees)", min_value=0, max_value=90, value=45, step=1,key='angle')

with col2:
    u = st.slider("Choose initial velocity (m/s)",min_value=10,max_value=100,value=30,step=10,key = "u")

st.write("y = ",angle+u)
