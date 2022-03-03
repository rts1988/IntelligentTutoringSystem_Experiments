import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
df = pd.read_csv('sample.csv')
G = nx.from_pandas_edgelist(df,'Node_from','Node_to')
st.set_option('deprecation.showPyplotGlobalUse', False)
import streamlit.components.v1 as components

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")

st.title("""
Hello I'm here!
""")
st.table(df)

# st.write("Showing networkx graph")
# st.write(G.edges)
# fig = plt.figure(figsize=(4,2))
# fig = nx.draw(G)
# st.pyplot(fig)

st.write("""
Showing pyvis network drawing
""")
# net = Network(height='700px', width = '500px',bgcolor='#222222', font_color='white')
# net.from_nx(G)
# net.repulsion(node_distance=420, central_gravity=0.33,
#                        spring_length=110, spring_strength=0.10,
#                        damping=0.95)
# Save and read graph as HTML file (on Streamlit Sharing)
try:
    # net.save_graph('trynet.html')
    HtmlFile = open('trynet.html', 'r', encoding='utf-8')
    #st.write(HtmlFile.read())
# Save and read graph as HTML file (locally)
except:
    st.write("Could not save graph or load")
# Load HTML file in HTML component for display on Streamlit page
components.html(HtmlFile.read(), height=500)
