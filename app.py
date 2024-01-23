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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    auc
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
classifier = None
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

if algorithm is None:
    st.error("Caution: Select Classification Algorithm")
    st.stop()
elif algorithm == "Naive Bayes":
    classifier = GaussianNB()
elif algorithm == "Logistic Regression":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        penalty_choice = st.selectbox("Regularization Type",
                                      ["L1", "L2", "Elastic Net", "None"],
                                      index=None)
        if penalty_choice == "L1":
            penalty = "l1"
        elif penalty_choice == "L2":
            penalty = "l2"
        elif penalty_choice == "Elastic Net":
            penalty = "elasticnet"
        else:
            penalty = None

        C = st.number_input("Inverse Regularization Strength (C)",
                            min_value=0.0,
                            value=None)
        params["C"] = C

        multi_class_choice = st.selectbox("Multi-class Classification",
                                          ["Auto",
                                           "One vs Rest",
                                           "Softmax"],
                                           index=None)
        if multi_class_choice == "Auto":
            multi_class = "auto"
        elif multi_class_choice == "One vs Rest":
            multi_class = "ovr"
        elif multi_class_choice == "Softmax":
            multi_class = "multinomial"
        else:
            multi_class = None
        params["multi_class"] = multi_class

    with column2:
        max_iter = st.slider("Maximum no. of Iterations",
                             min_value=0,
                             max_value=1000,
                             step=1,
                             value=None)
        params["max_iter"] = max_iter

        l1_ratio = st.slider("L1 Ratio (Elastic Net)",
                             min_value=0.0,
                             max_value=1.0,
                             step=0.01,
                             value=None)
        params["l1_ratio"] = l1_ratio

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Logistic Regression")
        st.stop()
    
    params["penalty"] = penalty
    params["random_state"] = random_state
    classifier = LogisticRegression(**params)
elif algorithm == "Support Vector Machine":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        kernel_choice = st.selectbox("Kernel",
                                     ["Linear",
                                      "RBF",
                                      "Polynomial",
                                      "Sigmoid"],
                                     index=None)
        if kernel_choice == "Linear":
            kernel = "linear"
        elif kernel_choice == "RBF":
            kernel = "rbf"
        elif kernel_choice == "Polynomial":
            kernel = "poly"
        elif kernel_choice == "Sigmoid":
            kernel = "sigmoid"
        else:
            kernel = None
        params["kernel"] = kernel

        C = st.number_input("Inverse Regularization Strength (C)",
                            min_value=0.0,
                            value=None)
        params["C"] = C

        gamma = st.number_input("Kernel Coefficient (gamma)",
                                min_value=0.0,
                                value=None)
        params["gamma"] = gamma
    
    with column2:
        degree = st.slider("Degree (Polynomial Kernel)",
                           min_value=0,
                           max_value=10,
                           step=1,
                           value=None)
        params["degree"] = degree

        coef0 = st.number_input("Kernel Coefficient (coef0)",
                                min_value=0.0,
                                value=None)
        params["coef0"] = coef0

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)
    
    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Support Vector Machine")
        st.stop()
    
    params["random_state"] = random_state
    classifier = SVC(**params)
elif algorithm == "Decision Tree":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        criterion_choice = st.selectbox("Criterion",
                                        ["Gini Impurity",
                                         "Entropy",
                                         "Log Loss"],
                                        index=None)
        if criterion_choice == "Entropy":
            criterion = "entropy"
        elif criterion_choice == "Gini Impurity":
            criterion = "gini"
        elif criterion_choice == "Log Loss":
            criterion = "log_loss"
        else:
            criterion = None
        params["criterion"] = criterion

        max_features_choice = st.selectbox("Maximum Features",
                                           ["All",
                                            "Square Root",
                                            "Log (base 2)"],
                                           index=None)
        if max_features_choice == "Log (base 2)":
            max_features = "log2"
        elif max_features_choice == "Square Root":
            max_features = "sqrt"
        else:
            max_features = None

        max_depth = st.slider("Maximum Depth",
                              min_value=1,
                              step=1,
                              value=None)
        
        min_samples_leaf = st.slider("Minimum samples for Leaf Node",
                                     min_value=1,
                                     max_value=50,
                                     step=1,
                                     value=None)
        params["min_samples_leaf"] = min_samples_leaf

    with column2:
        min_samples_split = st.number_input("Minimum samples to Split Node",
                                            min_value=0.01,
                                            max_value=1.0,
                                            step=0.01,
                                            value=None)
        params["min_samples_split"] = min_samples_split

        min_impurity_decrease = st.number_input("Minimum Impurity Decrease to Split Node",
                                                min_value=0.0,
                                                step=0.01,
                                                value=None)
        params["min_impurity_decrease"] = min_impurity_decrease

        ccp_alpha = st.number_input("Cost-complexity Pruning Parameter",
                                    min_value=0.0,
                                    step=0.01,
                                    value=None)
        params["ccp_alpha"] = ccp_alpha

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)
        
    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Decision Tree")
        st.stop()

    params["max_depth"] = max_depth
    params["max_features"] = max_features
    params["random_state"] = random_state
    classifier = DecisionTreeClassifier(**params)
elif algorithm == "Random Forest":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        n_estimators = st.slider("Maximum no. of Estimators (Ensemble Size)",
                                 min_value=1,
                                 max_value=1000,
                                 step=1,
                                 value=None)
        params["n_estimators"] = n_estimators

        criterion_choice = st.selectbox("Criterion",
                                        ["Gini Impurity",
                                         "Entropy",
                                         "Log Loss"],
                                        index=None)
        if criterion_choice == "Entropy":
            criterion = "entropy"
        elif criterion_choice == "Gini Impurity":
            criterion = "gini"
        elif criterion_choice == "Log Loss":
            criterion = "log_loss"
        else:
            criterion = None
        params["criterion"] = criterion

        max_features_choice = st.selectbox("Maximum Features",
                                           ["All",
                                            "Square Root",
                                            "Log (base 2)"],
                                           index=None)
        if max_features_choice == "Log (base 2)":
            max_features = "log2"
        elif max_features_choice == "Square Root":
            max_features = "sqrt"
        else:
            max_features = None
        
        max_depth = st.slider("Maximum Depth",
                              min_value=1,
                              step=1,
                              value=None)
        
        min_samples_leaf = st.slider("Minimum samples for Leaf Node",
                                     min_value=1,
                                     max_value=50,
                                     step=1,
                                     value=None)
        params["min_samples_leaf"] = min_samples_leaf
    
    with column2:
        min_samples_split = st.number_input("Minimum samples to Split Node",
                                            min_value=0.01,
                                            max_value=1.0,
                                            step=0.01,
                                            value=None)
        params["min_samples_split"] = min_samples_split

        min_impurity_decrease = st.number_input("Minimum Impurity Decrease to Split Node",
                                                min_value=0.0,
                                                step=0.01,
                                                value=None)
        params["min_impurity_decrease"] = min_impurity_decrease

        ccp_alpha = st.number_input("Cost-complexity Pruning Parameter",
                                    min_value=0.0,
                                    step=0.01,
                                    value=None)
        params["ccp_alpha"] = ccp_alpha

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Random Forest")
        st.stop()

    params["max_depth"] = max_depth
    params["max_features"] = max_features
    params["random_state"] = random_state
    classifier = RandomForestClassifier(**params)
elif algorithm == "Ada Boost":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        pass
    
    with column2:
        pass

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Ada Boost")
        st.stop()

    # params["max_depth"] = max_depth
    # params["max_features"] = max_features
    # params["random_state"] = random_state
    classifier = AdaBoostClassifier(**params)
elif algorithm == "Gradient Boosting":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        pass
    
    with column2:
        pass

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Gradient Boosting")
        st.stop()

    # params["max_depth"] = max_depth
    # params["max_features"] = max_features
    # params["random_state"] = random_state
    classifier = GradientBoostingClassifier(**params)
elif algorithm == "XG Boost":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        pass
    
    with column2:
        pass

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for XG Boost")
        st.stop()

    # params["max_depth"] = max_depth
    # params["max_features"] = max_features
    # params["random_state"] = random_state
    classifier = XGBClassifier(**params)

# decision-boundary display button
    
# model evaluation button

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