import streamlit as st
import pandas as pd

import streamlit.components.v1 as components

_custom_dataframe = components.declare_component(
    "custom_dataframe", url="http://localhost:3001",
)


def custom_dataframe(data, key=None):
    return _custom_dataframe(data=data, key=key, default=pd.DataFrame())


raw_data = {
    "First Name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    "Last Name": ["Miller", "Jacobson", "Ali", "Milner", "Smith"],
    "Age": [42, 52, 36, 24, 73],
}

df = pd.DataFrame(raw_data, columns=["First Name", "Last Name", "Age"])
returned_df = custom_dataframe(df)

if not returned_df.empty:
    st.table(returned_df)
