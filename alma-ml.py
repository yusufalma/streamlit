import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.sidebar.header('Machine Learning Application')

def iris_input():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

st.sidebar.subheader('Input parameters')
df = iris_input()

st.header('Multi-class Classification: Iris plants dataset')
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)
iris_predict = clf.predict(df)
iris_predict_proba = clf.predict_proba(df)

st.subheader('Class labels')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[iris_predict])

st.subheader('Prediction probability')
st.write(iris_predict_proba)

if st.checkbox('Show full description of the dataset'):
    st.write(iris.DESCR)
