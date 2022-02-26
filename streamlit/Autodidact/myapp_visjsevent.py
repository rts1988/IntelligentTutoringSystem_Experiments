import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
df = pd.read_csv('sample.csv')
G = nx.from_pandas_edgelist(df,'Node_from','Node_to')
st.set_option('deprecation.showPyplotGlobalUse', False)
import streamlit.components.v1 as components

st.title("""
Hello I'm here!
""")
st.table(df)

st.write("""
Showing visjs network drawing with on click event
""")

try:
    # net.save_graph('trynet.html')
    HtmlFile = open('trynet.html', 'r', encoding='utf-8')
# Save and read graph as HTML file (locally)
except:
    st.write("Could not save graph or load")
# Load HTML file in HTML component for display on Streamlit page
components.html(HtmlFile.read(), height=500,width=1000)
