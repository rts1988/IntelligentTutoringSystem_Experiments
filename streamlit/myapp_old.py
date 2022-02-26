import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import bz2
import pickle
import _pickle as cPickle

# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data


try:
 del uploaded_files
except:
 pass


uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
st.write("Files uploaded")

# for uploaded_file in uploaded_files:
#  concepts = decompress_pickle(uploaded_file)
#  st.write("filename:", uploaded_file.name)

filenames = [file.name for file in uploaded_files]

import pandas as pd
Agg_Conceptdata = pd.DataFrame()
All_Conceptdata = pd.DataFrame()
Agg_np_to_sent = dict()
Agg_sent_to_npflat = dict()
Agg_sent_to_phen = dict()
Agg_phen_to_sent = dict()
Agg_att_to_sent = dict()
Agg_sent_to_att = dict()
Agg_ins_to_sent = dict()
Agg_sent_to_ins = dict()
Agg_set_to_sent = dict()
Agg_np_to_forms = dict()
doc_to_np = dict()
np_to_doc = dict()
Agg_df = pd.DataFrame()
Agg_df = pd.DataFrame()
Agg_np_to_roles = dict()
Agg_sent_to_clt = dict()
Agg_sents = dict()
#Agg_sents_df = pd.DataFrame()
#Agg_docs_df = pd.DataFrame()
All_df = pd.DataFrame()

for uploaded_file in uploaded_files:
  concepts = decompress_pickle(uploaded_file)
  filename = uploaded_file.name
  #st.write("filename:", uploaded_file.name)

  Conceptdata = concepts['Conceptdata']
  sent_to_npflat = concepts['sent_to_npflat']
  np_to_sent = concepts['np_to_sent']
  np_to_forms = concepts['np_to_forms']
  sent_to_phen = concepts['sent_to_phen']
  phen_to_sent = concepts['phen_to_sent']
  sent_to_att = concepts['sent_to_att']
  att_to_sent = concepts['att_to_sent']
  att_to_sent = concepts['att_to_sent']
  ins_to_sent = concepts['ins_to_sent']
  sent_to_ins = concepts['sent_to_ins']
  set_to_sent = concepts['set_to_sent']
  sent_to_set = concepts['sent_to_set']
  np_to_roles = concepts['np_to_roles']
  sent_to_clt = concepts['sent_to_clt']
  sents = concepts['sents']
  df = concepts['df']

  Conceptdata['docname'] = filename
  Agg_Conceptdata = Agg_Conceptdata.append(Conceptdata,ignore_index=True)

  Agg_sent_to_clt[filename.replace(".pbz2","")] = sent_to_clt
  Agg_np_to_sent[filename.replace(".pbz2","")] = np_to_sent
  Agg_sents[filename.replace(".pbz2","")] = sents
  Agg_sent_to_npflat[filename.replace(".pbz2","")] = sent_to_npflat

  Agg_df = Agg_df.append(df,ignore_index=True)
  doc_to_np[filename] = list(np_to_sent.keys())

  for np in np_to_sent:
    # if np in Agg_np_to_sent:
    #   Agg_np_to_sent[np] = Agg_np_to_sent[np] + [(filename,s) for s in np_to_sent[np]]
    # else:
    #   Agg_np_to_sent[np] = [(filename,s) for s in np_to_sent[np]]

    if np in np_to_doc:
      np_to_doc[np] = np_to_doc[np] + [filename]
    else:
      np_to_doc[np] = [filename]

  for np in np_to_forms:
    if np in Agg_np_to_forms:
      Agg_np_to_forms[np] = Agg_np_to_forms[np] + np_to_forms[np]
    else:
      Agg_np_to_forms[np] = np_to_forms[np]

  for np in np_to_roles:
    if np in Agg_np_to_roles:
      Agg_np_to_roles[np] = Agg_np_to_roles[np] + np_to_roles[np]
    else:
      Agg_np_to_roles[np] = np_to_roles[np]

  for np in phen_to_sent:
    if np in Agg_phen_to_sent:
      Agg_phen_to_sent[np] = Agg_phen_to_sent[np] + [(filename,s) for s in phen_to_sent[np]]
    else:
      Agg_phen_to_sent[np] = [(filename,s) for s in phen_to_sent[np]]

  for np in att_to_sent:
    if np in Agg_att_to_sent:
      Agg_att_to_sent[np] = Agg_att_to_sent[np] + [(filename,s) for s in att_to_sent[np]]
    else:
      Agg_att_to_sent[np] =  [(filename,s) for s in att_to_sent[np]]

  for np in set_to_sent:
    if np in Agg_set_to_sent:
      Agg_set_to_sent[np] = Agg_set_to_sent[np] + [(filename,s) for s in set_to_sent[np]]
    else:
      Agg_set_to_sent[np] =  [(filename,s) for s in set_to_sent[np]]

  for np in ins_to_sent:
    if np in Agg_ins_to_sent:
      Agg_ins_to_sent[np] = Agg_ins_to_sent[np] + [(filename,s) for s in ins_to_sent[np]]
    else:
      Agg_ins_to_sent[np] =  [(filename,s) for s in ins_to_sent[np]]



st.write("""
Showing pyvis network drawing
""")
net = Network(height='700px', width = '500px',bgcolor='#222222', font_color='white')
net.from_nx(G)
net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.10,
                       damping=0.95)
# Save and read graph as HTML file (on Streamlit Sharing)
try:
    net.save_graph('pyvis_graph.html')
    HtmlFile = open('pyvis_graph.html', 'r', encoding='utf-8')
# Save and read graph as HTML file (locally)
except:
    st.write("Could not save graph or load")
# Load HTML file in HTML component for display on Streamlit page
components.html(HtmlFile.read(), height=435)
