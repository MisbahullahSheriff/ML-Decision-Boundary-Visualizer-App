import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)
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
            
### This app will analyze the Decision Boundary of a given Machine Learning Classifier for any given Dataset
""")

st.info("""
#### This app gives complete control to the User for:
- Selecting the Dataset to Analyze
- Selecting the Features for Modelling
- Selecting the steps to Preprocess the Dataset for Modelling
- Selecting a Machine Learning Classififcation Algorithm to use
- Tuning the Hyperparameters of the selected Learning Algorithm
""")

st.warning("""
#### Must See:
- The input file should be of '.csv' format
- The uploaded data must have well-defined column labels
- The uploaded data must have a discrete target variable (for classification)
- User must select 2 numeric continous features (for convenient visualization)
""")

# getting the data
file = st.sidebar.file_uploader("Upload Dataset (csv):")
if file is not None:
    df = pd.read_csv(file)
else:
    st.error("Caution: Upload Data", icon="🚨")
    st.stop()

st.success("Data successfully uploaded!")
with st.expander("View Uploaded Data:"):
    st.dataframe(df, hide_index=True, use_container_width=True)

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
    st.error("Caution: Select value for Feature 1")
    st.stop()
selected_cols.append(col1)

col2 = st.sidebar.selectbox("Select Feature 2",
                            columns,
                            index=None)
if col2 is None:
    st.error("Caution: Select value for Feature 2")
    st.stop()
elif col1 == col2:
    st.error("Caution: Feature 1 and Feature 2 must be distinct")
    st.stop()
selected_cols.append(col2)

target = st.sidebar.selectbox("Select Target Variable",
                              df.columns.to_list(),
                              index=None)
if target is None:
    st.error("Caution: Select value for Target Variable")
    st.stop()
elif (target == col1) or (target == col2):
    st.error("Caution: Target Variable must be distinct from Feature 1 and Feature 2")
    st.stop()

st.success("Features successfully retrieved!")

# splitting the data
X = df.loc[:, selected_cols]
y = df.loc[:, target].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

classes = label_encoder.classes_
n_classes = len(classes)
mapping = {
    i: label for i, label in enumerate(classes)
}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=7
)
test_indices = X_test.index

# displaying selected features
markers = ["o", "s", "^"]
colors = ["orange", "deepskyblue", "forestgreen"]

column1, column2 = st.columns(2)

with column1:
    with st.expander("View Filtered Data:"):
        st.dataframe(df.loc[:, selected_cols],
                     use_container_width=True,
                     hide_index=True)

with column2:
    fig, ax = plt.subplots()
    for cls, marker, color in zip(np.unique(y),
                                  markers[:n_classes],
                                  colors[:n_classes]):
        subset = (y == cls)
        ax.scatter(
            X.values[subset, 0],
            X.values[subset, 1],
            marker=marker,
            c=color,
            edgecolors="black",
            label=f"{mapping[cls]}"
        )
    ax.set(xlabel=selected_cols[0],
           ylabel=selected_cols[1])
    ax.set_title("Filtered Data",
                 fontweight="bold",
                 fontsize=12)
    ax.legend(loc="best")
    st.pyplot(fig)

# imputation strategy
imputation_choice = st.sidebar.selectbox("Select Imputation Strategy",
                                         ["Mean", "Median", "Most Frequent", "Constant Value"],
                                         index=None)
fill_value_choice = None

if imputation_choice is None:
    st.error("Caution: Select Imputation Strategy for handling Missing Values")
    st.stop()
elif imputation_choice == "Mean":
    strategy = "mean"
elif imputation_choice == "Median":
    strategy = "median"
elif imputation_choice == "Most Frequent":
    strategy = "most_frequent"
elif imputation_choice == "Constant Value":
    strategy = "constant"
    fill_value_choice = st.sidebar.number_input("Enter Constant Value", value=None)
    if fill_value_choice is None:
        st.error("Caution: Select Constant Value for Imputation")
        st.stop()

imputer = SimpleImputer(strategy=strategy, fill_value=fill_value_choice)

# feature scaling strategy
scaling_strategy = st.sidebar.selectbox("Select Feature Scaling Strategy",
                                        ["Standardization",
                                         "Normalization",
                                         "No Scaling"],
                                        index=None)

if scaling_strategy is None:
    st.error("Caution: Select value for Feature Scaling Strategy")
    st.stop()
elif scaling_strategy == "Standardization":
    scaler = StandardScaler()
elif scaling_strategy == "Normalization":
    scaler = MinMaxScaler()
else:
    scaler = None

# preprocessing data
preprocessor = Pipeline(steps=[
    ("imputer", imputer),
    ("scaler", scaler)
])

X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

st.success("Data successfully preprocessed!")

# selecting algorithm
algorithm = st.selectbox("Select Classification Algorithm",
                         ["Naive Bayes",
                          "Logistic Regression",
                          "Support Vector Machine",
                          "Decision Tree",
                          "Random Forest",
                          "Ada Boost",
                          "Gradient Boosting",
                          "XG Boost"],
                         index=None)

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