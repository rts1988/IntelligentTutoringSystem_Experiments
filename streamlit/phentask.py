import streamlit as st
import pandas as pd
import copy


#st.write(list(st.session_state.keys()))
if 'phendf' not in st.session_state:
  st.session_state['phendf'] = pd.DataFrame()
  #st.session_state['phendf'].columns = ['phen1','rel','phen2']
if 'counter' not in st.session_state:
  st.session_state['counter'] = 0
if 'save_res' not in st.session_state:
  st.session_state['save_res'] = False




phenlist = ["volcano","earthquake","light","heat"]
rel_list = ["promotes","inhibits","follows","is followed by","competes with"]

if 'phen1' not in st.session_state:
  st.session_state.phen1 = phenlist[0]

if 'phen2' not in st.session_state:
  st.session_state.phen2 = phenlist[1]

if 'rel' not in st.session_state:
  st.session_state.rel = rel_list[0]


col1,col2,col3,col4 = st.columns(4)

phenlist_removedphen2 = copy.deepcopy(phenlist)
phenlist_removedphen2.remove(st.session_state.phen2)
#st.write(phenlist_removedphen2)
with col1:
  st.session_state.phen1 = st.selectbox(label = "Choose phenomenon 1",options = phenlist_removedphen2,index=0)

with col2:
  st.session_state.rel = st.selectbox(label = "Choose phenomenon",options = rel_list,index=0)

phenlist_removedphen1 = copy.deepcopy(phenlist)
phenlist_removedphen1.remove(st.session_state.phen1)
#st.write(phenlist_removedphen1)
with col3:
  st.session_state.phen2 = st.selectbox(label = "Choose phenomenon 2",options = phenlist_removedphen1,index=0)

st.write(st.session_state.phen1,st.session_state.rel,st.session_state.phen2)

def onclicksave():
  st.session_state['phendf'] = st.session_state['phendf'].append({'phen1':st.session_state.phen1,'rel':st.session_state.rel,'phen2':st.session_state.phen2},ignore_index=True)
  st.table(st.session_state['phendf'])
  st.session_state['counter'] +=1
  return st.session_state


# with col4:
#   st.button(label = "+",on_click = onclicksave)
with col4:
  st.session_state['save_res'] = st.button(label = "+")


st.write(st.session_state.save_res)
if st.session_state.save_res:
    #onclicksave()
  st.session_state['phendf'] = st.session_state['phendf'].append({'phen1':st.session_state.phen1,'rel':st.session_state.rel,'phen2':st.session_state.phen2},ignore_index=True)
st.table(st.session_state['phendf'])
st.session_state['counter'] +=1





st.write(st.session_state['counter'])
