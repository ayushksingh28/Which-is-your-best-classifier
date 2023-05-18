import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


st.title("Streamlit Example")

st.write("""
# Explore different types of classifiers
# Which one is the best ?
""")
         
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "Random Forest", "SVM"))
st.write(classifier_name)

def get_datasets(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target

    return X,y

X,y = get_datasets(dataset_name)

st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

def add_params_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.00)
        params["C"] = C
    else:
        Depth = st.sidebar.slider("Depth", 2, 15)
        N_estimators = st.sidebar.slider("N_estmators", 1, 100)
        params["Depth"] = Depth
        params["N_estimators"] = N_estimators
    return params

params = add_params_ui(classifier_name)    

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name =="SVM":
        clf = SVC(C= params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["N_estimators"],
                                     max_depth = params["Depth"])

    return clf

clf = get_classifier(classifier_name, params)            

#Classification


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier Name = {classifier_name}")
st.write(f"Accuracy = {acc}")

#PLOT

pca = PCA(2)
X_projected  = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
