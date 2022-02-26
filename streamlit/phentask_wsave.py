import streamlit as st
import pandas as pd
import copy
import pickle
import matplotlib.pyplot as plt
import networkx as nx

phenlist = ["volcano","earthquake","light","heat"]
rel_list = ["promotes","inhibits","follows","is followed by","competes with"]

if 'save_res' not in st.session_state:
  st.session_state.save_res = False

try:

  #phendf = pd.read_csv('phendf.csv')
  file0 = open('phendf.pkl',"rb")
  phendf = pickle.load(file0)
  file0.close()
  #phendf = phendf.loc[:,['phen1','rel','phen2']]
except:
  st.write('creating new phendf')
  phendf = pd.DataFrame()

try:
  #current_vals = pd.read_csv('currentvals.csv')
  file1 = open('current_vals.pkl',"rb")
  current_vals = pickle.load(file1)
  file1.close()
except:
  st.write('creating new current vals')
  current_vals = dict()
  current_vals['phen1'] = phenlist[0]
  current_vals['rel'] = rel_list[0]
  current_vals['phen2'] = phenlist[1]

col1,col2,col3,col4 = st.columns(4)

with col1:
  phen1 = st.selectbox(label = "Choose phenomenon 1",options = phenlist,index = phenlist.index(current_vals['phen1']))

with col2:
  rel = st.selectbox(label = "Choose relationship",options = rel_list,index=rel_list.index(current_vals['rel']))


with col3:
  phen2 = st.selectbox(label = "Choose phenomenon 2",options = phenlist,index=phenlist.index(current_vals['phen2']))


current_vals['phen1'] = phen1
current_vals['rel'] = rel
current_vals['phen2'] = phen2
file2 = open("current_vals.pkl", "wb")
pickle.dump(current_vals,file2)
file2.close()

st.write(phen1,rel,phen2)

# with col4:
#   st.button(label = "+",on_click = onclicksave)
with col4:
  st.session_state.save_res = st.button(label = "+")


st.write(st.session_state.save_res)

if st.session_state.save_res:
    #onclicksave()
  phendf = phendf.append({'phen1':phen1,'rel':rel,'phen2':phen2},ignore_index=True)
  phendf = phendf.drop_duplicates()
  file3 = open('phendf.pkl',"wb")
  pickle.dump(phendf,file3)
  file3.close()



if phendf.shape[0]>0:
  fig, ax = plt.subplots()
  G = nx.from_pandas_edgelist(phendf, 'phen1', 'phen2',['rel'],create_using=nx.DiGraph())
  pos = nx.spectral_layout(G)
  nx.draw(G, pos, with_labels=True)
  edge_labels = nx.get_edge_attributes(G,'rel')
  nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
  st.pyplot(fig)
  st.table(phendf)
