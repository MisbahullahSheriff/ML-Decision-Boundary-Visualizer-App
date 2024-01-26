import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
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
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix
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
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from imblearn.metrics import geometric_mean_score
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

# credits
st.markdown("""
---
#### Created By - Mohammed Misbahullah Sheriff
""")

github_url = "https://github.com/MisbahullahSheriff"
linkedin_url = "https://www.linkedin.com/in/mohammed-misbahullah-sheriff/"

st.markdown(
    f'<div style="display: flex;">'
    f'<a href="{linkedin_url}" style="margin-right: 10px; padding: 8px 20px;background-color: #1e6ed6; color: white; text-align: center; text-decoration:none; font-size: 16px; border-radius: 4px;">LinkedIN</a>'
    f'<a href="{github_url}" style="padding: 8px 20px; background-color: #2a8503;color: white; text-align: center; text-decoration: none; font-size: 16px;border-radius: 4px;">GitHub</a>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# web-app guidelines
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
- The target variable must contain no more than 10 unique labels
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
X = (
    df
    .loc[:, selected_cols]
    .assign(**{
        f"{col}": pd.to_numeric(df[col], errors="coerce") for col in selected_cols
    })
)
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
color_mapping = {
    "Red": "#b50600",
    "Blue": "#02449c",
    "Green": "#02690c",
    "Purple": "#710299",
    "Orange": "#f25e02",
    "Jade": "#02ad8e",
    "Dark Pink": "#ab023a",
    "Dark Green": "#064201",
    "Violet": "#2a0142",
    "Dark Brown": "#420b01",
}

column1, column2 = st.columns(2)

with column1:
    with st.expander("View Filtered Data:"):
        st.dataframe(X,
                     use_container_width=True,
                     hide_index=True)

with column2:
    fig, ax = plt.subplots()
    for cls, color in zip(np.unique(y),
                          list(color_mapping.values())[:n_classes]):
        subset = (y == cls)
        ax.scatter(
            X.values[subset, 0],
            X.values[subset, 1],
            marker="o",
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
                          "XG Boost",
                          "Neural Network"],
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
                            min_value=1e-4,
                            step=1e-4,
                            format="%.4f",
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
                             min_value=1,
                             max_value=1000,
                             step=1,
                             value=None)
        params["max_iter"] = max_iter

        l1_ratio = st.slider("L1 Ratio (Elastic Net)",
                             min_value=0.0,
                             max_value=1.0,
                             step=1e-4,
                             format="%.4f",
                             value=None)

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Logistic Regression")
        st.stop()
    
    params["penalty"] = penalty
    params["l1_ratio"] = l1_ratio
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
                            min_value=1e-4,
                            step=1e-4,
                            format="%.4f",
                            value=None)
        params["C"] = C

        gamma = st.number_input("Kernel Coefficient (gamma)",
                                min_value=0.0,
                                step=1e-4,
                                format="%.4f")
    
    with column2:
        degree = st.slider("Degree (Polynomial Kernel)",
                           min_value=0,
                           max_value=10,
                           step=1,
                           value=None)

        coef0 = st.number_input("Kernel Coefficient (coef0)",
                                min_value=-9999.0,
                                step=1e-4,
                                format="%.4f",
                                value=0.0)

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)
    
    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Support Vector Machine")
        st.stop()
    
    params["coef0"] = coef0
    params["gamma"] = gamma
    params["degree"] = degree
    params["probability"] = True
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
        min_samples_split = st.slider("Fraction of minimum samples to Split Node",
                                      min_value=0.01,
                                      max_value=1.0,
                                      step=0.01,
                                      value=None)
        params["min_samples_split"] = min_samples_split

        min_impurity_decrease = st.number_input("Minimum Impurity Decrease to Split Node",
                                                min_value=0.0,
                                                step=1e-4,
                                                format="%.4f")

        ccp_alpha = st.number_input("Cost-complexity Pruning Parameter",
                                    min_value=0.0,
                                    step=1e-4,
                                    format="%.4f")

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)
        
    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Decision Tree")
        st.stop()

    params["max_depth"] = max_depth
    params["ccp_alpha"] = ccp_alpha
    params["max_features"] = max_features
    params["random_state"] = random_state
    params["min_impurity_decrease"] = min_impurity_decrease
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
        min_samples_split = st.slider("Fraction of minimum samples to Split Node",
                                      min_value=0.01,
                                      max_value=1.0,
                                      step=0.01,
                                      value=None)
        params["min_samples_split"] = min_samples_split

        min_impurity_decrease = st.number_input("Minimum Impurity Decrease to Split Node",
                                                min_value=0.0,
                                                step=1e-4,
                                                format="%.4f")

        ccp_alpha = st.number_input("Cost-complexity Pruning Parameter",
                                    min_value=0.0,
                                    step=1e-4,
                                    format="%.4f")

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Random Forest")
        st.stop()

    params["max_depth"] = max_depth
    params["ccp_alpha"] = ccp_alpha
    params["max_features"] = max_features
    params["random_state"] = random_state
    params["min_impurity_decrease"] = min_impurity_decrease
    classifier = RandomForestClassifier(**params)
elif algorithm == "Ada Boost":
    params = dict()

    n_estimators = st.slider("Maximum no. of Estimators (Ensemble Size)",
                             min_value=1,
                             max_value=1000,
                             step=1,
                             value=None)
    params["n_estimators"] = n_estimators

    algorithm_choice = st.selectbox("Algorithm",
                                    ["SAMME", "SAMME.R"],
                                    index=None)
    params["algorithm"] = algorithm_choice

    learning_rate = st.number_input("Learning Rate",
                                    min_value=1e-4,
                                    step=1e-4,
                                    format="%.4f",
                                    value=None)
    params["learning_rate"] = learning_rate

    random_state = st.number_input("Random State",
                                   min_value=0,
                                   step=1,
                                   value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Ada Boost")
        st.stop()

    params["random_state"] = random_state
    classifier = AdaBoostClassifier(**params)
elif algorithm == "Gradient Boosting":
    st.warning("""
#### Disclaimer:
- For multi-class classification, select `Log Loss` for parameter `Loss Function`
- The `Exponential` loss function only works for binary classification problems
""")
    
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        loss_choice = st.selectbox("Loss Function",
                                   ["Log Loss",
                                    "Exponential"],
                                   help="Use 'Log Loss' for multi-class classififcation",
                                   index=None)
        if loss_choice == "Log Loss":
            loss = "log_loss"
        elif loss_choice == "Exponential":
            loss = "exponential"
        else:
            loss = None
        params["loss"] = loss

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
        
        n_estimators = st.slider("Maximum no. of Estimators (Ensemble Size)",
                                 min_value=1,
                                 max_value=1000,
                                 step=1,
                                 value=None)
        params["n_estimators"] = n_estimators

        max_depth = st.slider("Maximum Depth",
                              min_value=1,
                              max_value=10,
                              step=1,
                              value=None)

        subsample = st.slider("Fraction of sample size for each Base Estimator",
                              min_value=0.01,
                              max_value=1.0,
                              step=0.01,
                              value=None)
        params["subsample"] = subsample
        
        min_samples_leaf = st.slider("Minimum samples for Leaf Node",
                                     min_value=1,
                                     max_value=50,
                                     step=1,
                                     value=None)
        params["min_samples_leaf"] = min_samples_leaf

        n_iter_no_change = st.slider("Number of Iterations for Early Stopping",
                                     min_value=1,
                                     step=1,
                                     value=None)
    
    with column2:
        min_samples_split = st.slider("Fraction of minimum samples to Split Node",
                                      min_value=0.01,
                                      max_value=1.0,
                                      step=0.01,
                                      value=None)
        params["min_samples_split"] = min_samples_split

        learning_rate = st.number_input("Learning Rate",
                                        min_value=1e-4,
                                        step=1e-4,
                                        format="%.4f",
                                        value=None)
        params["learning_rate"] = learning_rate

        min_impurity_decrease = st.number_input("Minimum Impurity Decrease to Split Node",
                                                min_value=0.0,
                                                step=1e-4,
                                                format="%.4f")

        tol = st.number_input("Tolerance for Early Stopping",
                              min_value=1e-4,
                              step=1e-4,
                              format="%.4f",
                              value=None)
        params["tol"] = tol

        ccp_alpha = st.number_input("Cost-complexity Pruning Parameter",
                                    min_value=0.0,
                                    step=1e-4,
                                    format="%.4f")

        random_state = st.number_input("Random State",
                                   min_value=0,
                                   step=1,
                                   value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Gradient Boosting")
        st.stop()

    params["max_depth"] = max_depth
    params["ccp_alpha"] = ccp_alpha
    params["max_features"] = max_features
    params["random_state"] = random_state
    params["n_iter_no_change"] = n_iter_no_change
    params["min_impurity_decrease"] = min_impurity_decrease
    classifier = GradientBoostingClassifier(**params)
elif algorithm == "XG Boost":
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        n_estimators = st.slider("Maximum no. of Estimators (Ensemble Size)",
                                 min_value=1,
                                 max_value=1000,
                                 step=1,
                                 value=None)
        params["n_estimators"] = n_estimators

        max_depth = st.slider("Maximum Depth",
                              min_value=1,
                              max_value=10,
                              step=1,
                              value=None)

        eta = st.slider("Step-size Shrinkage",
                        min_value=0.0,
                        max_value=1.0,
                        step=1e-4,
                        value=None)
        
        subsample = st.slider("Fraction of sample size for each Base Estimator",
                              min_value=0.01,
                              max_value=1.0,
                              step=0.01,
                              value=None)
        params["subsample"] = subsample

        colsample_bytree = st.slider("Fraction of features for each Tree",
                                     min_value=0.01,
                                     max_value=1.0,
                                     step=0.01,
                                     value=None)
        params["colsample_bytree"] = colsample_bytree

        colsample_bylevel = st.slider("Fraction of features for each Tree Level",
                                      min_value=0.01,
                                      max_value=1.0,
                                      step=0.01,
                                      value=None)
        params["colsample_bylevel"] = colsample_bylevel
    
    with column2:
        gamma = st.number_input("Minimum Loss Reduction for Split",
                                min_value=0.0,
                                step=1e-4,
                                format="%.4f",
                                value=None)
        
        alpha = st.number_input("L1 Regularization",
                                min_value=0.0,
                                step=1e-4,
                                format="%.4f",
                                value=None)
        
        lambda_ = st.number_input("L2 Regularization",
                                  min_value=0.0,
                                  step=1e-4,
                                  format="%.4f",
                                  value=None)
        
        min_child_weight = st.number_input("Minimum sum of Instance Weights in each Tree",
                                           min_value=0.0,
                                           step=1e-4,
                                           format="%.4f",
                                           value=None)
        
        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for XG Boost")
        st.stop()

    params["eta"] = eta
    params["gamma"] = gamma
    params["alpha"] = alpha
    params["lambda"] = lambda_
    params["max_depth"] = max_depth
    params["min_child_weight"] = min_child_weight
    params["random_state"] = random_state
    classifier = XGBClassifier(**params)
else:
    st.warning("""
#### Disclaimer:
- Provide comma-separated integers for the parameter `Size of Hidden Layer(s)`
    - Ex1: 10, 1
    - Ex2: 10, 5, 2
- `Size of Hidden Layer(s)` will not accept any character(s) apart from numbers and comma
""")
    
    params = dict()
    column1, column2 = st.columns(2)

    with column1:
        layers_text = st.text_input("Size of Hidden Layer(s)",
                                    value=None,
                                    placeholder="Provide input in appropriate format",
                                    help="Check the Disclaimer above")
        if layers_text is None:
            params["hidden_layer_sizes"] = None
        else:
            pattern = re.compile(r"[^0-9,]")
            temp = layers_text.replace(" ", "")
            search = re.search(pattern, temp)
            if search is None:
                temp = temp.split(",")
                layers = np.array([int(size) for size in temp if size != ""])
                params["hidden_layer_sizes"] = True
            else:
                st.error("Caution: Provide appropriate input for Size of Hidden Layer(s)")
                st.stop()
        
        activation_choice = st.selectbox("Activation Function",
                                         ["Identity",
                                          "Sigmoid",
                                          "Tan-H",
                                          "ReLU"],
                                         index=None)
        if activation_choice == "Uniform":
            activation = "identity"
        elif activation_choice == "Sigmoid":
            activation = "logistic"
        elif activation_choice == "Tan-H":
            activation = "tanh"
        elif activation_choice == "ReLU":
            activation = "relu"
        else:
            activation = None
        params["activation"] = activation

        solver_choice = st.selectbox("Optimizer",
                                     ["Adam",
                                      "LBFGS",
                                      "Stochastic Gradient Descent"],
                                     index=None)
        if solver_choice == "Adam":
            solver = "adam"
        elif solver_choice == "LBFGS":
            solver = "lbfgs"
        elif solver_choice == "Stochastic Gradient Descent":
            solver = "sgd"
        else:
            solver = None
        params["solver"] = solver

        max_iter = st.slider("Maximum no. of Iterations for Convergence",
                             min_value=1,
                             max_value=1000,
                             step=1,
                             value=None)
        params["max_iter"] = max_iter

        n_iter_no_change = st.slider("Number of Iterations for Early Stopping",
                                     min_value=1,
                                     step=1,
                                     value=None)

    with column2:
        learning_rate = st.number_input("Learning Rate",
                                    min_value=1e-4,
                                    step=1e-4,
                                    format="%.4f",
                                    value=None)
        params["learning_rate_init"] = learning_rate

        alpha = st.number_input("L2 Regularization",
                                min_value=0.0,
                                step=1e-4,
                                format="%.4f")
        
        tol = st.number_input("Tolerance for Early Stopping",
                              min_value=1e-4,
                              step=1e-4,
                              format="%.4f",
                              value=None)
        params["tol"] = tol

        random_state = st.number_input("Random State",
                                       min_value=0,
                                       step=1,
                                       value=None)

    if not all(params.values()):
        st.error("Caution: Select hyperparameters for Neural Network")
        st.stop()

    params["alpha"] = alpha
    params["hidden_layer_sizes"] = layers
    params["random_state"] = random_state
    params["n_iter_no_change"] = n_iter_no_change
    classifier = MLPClassifier(**params)

# training classifier button
if st.button("Train Classifier", use_container_width=True):
    st.session_state["classifier"] = classifier.fit(X_train_pre, y_train)
    st.success(f"{algorithm} Classifier successully trained!")

# getting marker
marker_mapping = {
    "Star": "*",
    "Circle": "o",
    "Square": "s",
    "Triangle": "^",
    "Diamond": "D",
    "Pentagon": "p",
    "Octagon": "8"
}
marker_choice = st.selectbox("Select Marker",
                             marker_mapping.keys(),
                             index=None)
if marker_choice is None:
    st.error("Caution: Select value for Marker")
    st.stop()
else:
    marker = marker_mapping[marker_choice]

# getting colors for unique class labels
selected_colors = st.multiselect(f"Select {n_classes} Colors for class labels",
                                 color_mapping.keys(),
                                 default=None)
if selected_colors is None:
    st.error("Caution: Select colors for class labels")
    st.stop()
elif len(selected_colors) != n_classes:
    st.error(f"Caution: Select exactly {n_classes} colors")
    st.stop()
else:
    color_codes = [color_mapping[color] for color in selected_colors]

# decision-boundary display button
def plot_data():
    # concatenating train and test data for plot
    X_total = np.vstack([X_train_pre, X_test_pre])
    y_total = np.hstack([y_train, y_test])

    # plotting entire data
    for cls, color in zip(np.unique(y_total), color_codes):
        subset = (y_total == cls)
        plt.scatter(
            X_total[subset, 0],
            X_total[subset, 1],
            s=45,
            color=color,
            marker=marker,
            label=mapping[cls]
        )
    
    # marking test data
    plt.scatter(
        X_test_pre[:, 0],
        X_test_pre[:, 1],
        s=180,
        color="none",
        marker="o",
        edgecolors="black",
        label="Test Data"
    )

def plot_boundary():
    min_values = np.min(X_train_pre, axis=0) - 0.5
    max_values = np.max(X_train_pre, axis=0) + 0.5
    xx, yy = np.meshgrid(
        np.linspace(min_values[0], max_values[0], 100),
        np.linspace(min_values[1], max_values[1], 100)
    )
    X_new = np.c_[xx.ravel(), yy.ravel()]
    try:
        y_new_pred = st.session_state["classifier"].predict(X_new).reshape(xx.shape)
    except:
        st.error("Caution: Classifier is not trained yet")
        st.stop()

    cmap = ListedColormap(color_codes)
    plt.contourf(xx, yy, y_new_pred, cmap=cmap, alpha=0.5)
    plt.contour(xx, yy, y_new_pred, colors="black", linewidths=0.5)

if st.button("Show Decision Boundary / Evaluate Classifier", use_container_width=True):
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_boundary()
    plot_data()
    ax.set_xlabel(f"{selected_cols[0]}", fontweight="bold", fontsize=12)
    ax.set_ylabel(f"{selected_cols[1]}", fontweight="bold", fontsize=12)
    ax.set_title(f"Decision Boundary of {algorithm}", fontweight="bold", fontsize=15)
    ax.legend(loc="upper left",
              bbox_to_anchor=(1.02, 1),
              title="Class Labels",
              title_fontproperties=dict(weight="bold", size=15),
              fontsize=12)
    st.pyplot(fig)
    
    # model evaluation
    column1, column2 = st.columns(2)

    try:
        y_pred = st.session_state["classifier"].predict(X_test_pre)
    except:
        st.error("Caution: Classifier is not trained yet")
        st.stop()

    # confusion-matrix
    with column1:
        fig, ax = plt.subplots(figsize=(6, 4))
        hm = sns.heatmap(
            confusion_matrix(y_test, y_pred),
            cmap="Blues",
            vmin=0,
            annot=True,
            square=True,
            linewidths=1.5,
            linecolor="white",
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title(f"Confusion Matrix of {algorithm}",
                     fontweight="bold")
        st.pyplot(fig)

    # metrics
    with column2:
        sub_column1, sub_column2 = st.columns(2)

        with sub_column1:
            # accuracy
            acc = f"{accuracy_score(y_test, y_pred):.2f}"
            st.metric(label="Accuracy", value=acc)

            # g-mean
            gmean = f"{geometric_mean_score(y_test, y_pred):.2f}"
            st.metric(label="G-Mean", value=gmean)

            # mcc
            mcc = f"{matthews_corrcoef(y_test, y_pred):.2f}"
            st.metric(label="Matthew's CC", value=mcc)

