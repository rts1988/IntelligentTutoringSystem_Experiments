import streamlit as st
import pandas as pd
import copy
import pickle
import matplotlib.pyplot as plt
import networkx as nx

setlist = ["volcano","earthquake","light","heat"]
rel_list = ["is a type of","is a superset of","is identical to","is disjoint from","overlaps with"]

if 'save_res' not in st.session_state:
  st.session_state.save_res = False

try:

  #phendf = pd.read_csv('phendf.csv')
  file0 = open('setdf.pkl',"rb")
  setdf = pickle.load(file0)
  file0.close()
  #phendf = phendf.loc[:,['phen1','rel','phen2']]
except:
  st.write('creating new setdf')
  setdf = pd.DataFrame()

try:
  #current_vals = pd.read_csv('currentvals.csv')
  file1 = open('current_vals_set.pkl',"rb")
  current_vals = pickle.load(file1)
  file1.close()
except:
  st.write('creating new current vals')
  current_vals = dict()
  current_vals['set1'] = setlist[0]
  current_vals['rel'] = rel_list[0]
  current_vals['set2'] = setlist[1]

if current_vals['set1'] not in setlist:
    current_vals['set1'] = setlist[0]
if current_vals['set2'] not in setlist:
    current_vals['set2'] = setlist[1]

col1,col2,col3,col4 = st.columns(4)

with col1:
  set1 = st.selectbox(label = "Choose set 1",options = setlist,index = setlist.index(current_vals['set1']))

with col2:
  rel = st.selectbox(label = "Choose relationship",options = rel_list,index=rel_list.index(current_vals['rel']))


with col3:
  set2 = st.selectbox(label = "Choose phenomenon 2",options = setlist,index=setlist.index(current_vals['set2']))


current_vals['set1'] = set1
current_vals['rel'] = rel
current_vals['set2'] = set2
file2 = open("current_vals_set.pkl", "wb")
pickle.dump(current_vals,file2)
file2.close()

st.write(set1,rel,set2)

# with col4:
#   st.button(label = "+",on_click = onclicksave)
with col4:
  st.session_state.save_res = st.button(label = "+")


st.write(st.session_state.save_res)

if st.session_state.save_res:
    #onclicksave()
  setdf = setdf.append({'set1':set1,'rel':rel,'set2':set2},ignore_index=True)
  setdf = setdf.drop_duplicates()
  file3 = open('setdf.pkl',"wb")
  pickle.dump(setdf,file3)
  file3.close()



if setdf.shape[0]>0:
  fig, ax = plt.subplots()
  G = nx.from_pandas_edgelist(setdf, 'set1', 'set2',['rel'],create_using=nx.DiGraph())
  pos = nx.planar_layout(G)
  nx.draw(G, pos, with_labels=True)
  edge_labels = nx.get_edge_attributes(G,'rel')
  nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
  st.pyplot(fig)
  st.table(setdf)
