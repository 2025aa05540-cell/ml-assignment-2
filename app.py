import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Assignment", layout="wide")

st.title("ML Assignment - Classification Models")
st.write("Upload your dataset and evaluate classification models.")

uploaded_file = st.file_uploader("Upload CSV file (test data)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())
else:
    st.info("Upload a CSV file to continue.")
