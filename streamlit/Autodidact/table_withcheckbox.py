import streamlit as st
import pandas as pd
import pickle
import _pickle as cPickle

file0 = open("setdf.pkl","rb")
setdf = pickle.load(file0)
file0.close()

num_columns = setdf.shape[1]

st.dataframe(setdf)

# checkboxdict = dict()
# for i,row in setdf.iterrows():
#     coltuples = st.columns(num_columns+1)
#     with coltuples[0]:
#         checkboxdict[i]= st.checkbox("",key=i)
#     for col in range(num_columns):
#         with coltuples[col]:
#             st.write(setdf.iloc[i,col])


#for i,row in phendf.iterrows():
