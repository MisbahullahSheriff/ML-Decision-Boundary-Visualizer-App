import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

st.set_page_config(
    page_icon="📈",
    page_title="ML Decision Boundary Visualizer",
    layout="wide",
)

# page header
st.markdown("""
# Decision Boundary Visualizer - Web App
            
### This app will plot the Decision Boundary of a given Machine Learning Classifier for any given dataset
""")

st.warning("""
#### Must See:
- User must upload files of '.csv' format
- User must ensure the uploaded data has well-defined column labels
- User must select 2 numeric continous features (for convenient visualization)
""")

# getting the data
file = st.sidebar.file_uploader("Upload Dataset (csv):")
if file is not None:
    df = pd.read_csv(file)
else:
    st.warning("Caution: Upload Data")
    st.stop()

st.success("Data successfully uploaded!")
with st.expander("View Uploaded Data:"):
    st.dataframe(df)

# getting the features
selected_cols = []
columns = (
    df
    .select_dtypes(include="number")
    .columns
    .to_list()
)

col1 = st.sidebar.selectbox("Select Feature 1",
                            columns,
                            index=None)
if col1 is None:
    st.warning("Caution: Select value for Feature 1")
    st.stop()
selected_cols.append(col1)

col2 = st.sidebar.selectbox("Select Feature 2",
                            columns,
                            index=None)
if col2 is None:
    st.warning("Caution: Select value for Feature 2")
    st.stop()
elif col1 == col2:
    st.error("Caution: Feature 1 and Feature 2 must be distinct")
    st.stop()
selected_cols.append(col2)

st.success("Features successfully retrieved!")

with st.expander("View Filtered Data:"):
    st.dataframe(df.loc[:, selected_cols])

# page footer
st.markdown("""
---
#### Created By - Mohammed Misbahullah Sheriff
""")

github_url = "https://github.com/MisbahullahSheriff"
linkedin_url = "https://www.linkedin.com/in/mohammed-misbahullah-sheriff/"

st.markdown(
    f'<div style="display: flex;">'
    f'<a href="{linkedin_url}" style="margin-right: 10px; padding: 8px 20px; background-color: #1e6ed6; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">LinkedIN</a>'
    f'<a href="{github_url}" style="padding: 8px 20px; background-color: #2a8503; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">GitHub</a>'
    f'</div>',
    unsafe_allow_html=True
)