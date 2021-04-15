import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title('ALMA: A Lesson to Machine_learning and Artificial_intelligence')
# st.sidebar.header('Machine Learning Application')

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

def diabetes_input():
    age = st.sidebar.slider('Age', -0.107226, 0.110727, 0.0)
    sex = st.sidebar.slider('Sex', -0.044642, 0.050680, 0.0)
    bmi = st.sidebar.slider('Body Mass Index', -0.090275, 0.170555, 0.0)
    bp = st.sidebar.slider('Average blood pressure', -0.112400, 0.132044, 0.0)
    s1 = st.sidebar.slider('T-Cells \(a type of white blood cells\)', -0.126781, 0.153914, 0.0)
    s2 = st.sidebar.slider('Low-Density Lipoproteins \(LDL\)', -0.115613, 0.198788, 0.0)
    s3 = st.sidebar.slider('High-Density Lipoproteins \(HDL\)', -0.102307, 0.181179, 0.0)
    s4 = st.sidebar.slider('Thyroid Stimulating Hormone', -0.076395, 0.185234, 0.0)
    s5 = st.sidebar.slider('Lamotrigine', -0.126097, 0.133599, 0.0)
    s6 = st.sidebar.slider('Blood sugar level', -0.137767, 0.135612, 0.0)
    data = {'age': age, 'sex': sex, 'bmi': bmi, 'bp': bp, 's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6}
    features = pd.DataFrame(data, index=[0])
    return features

# Choose between regression and classification, NLP, computer vision
app = st.sidebar.radio("Choose application",('Classification', 'Regression'))
if app == 'Classification':
    st.sidebar.subheader('Input parameters')
    df = iris_input()
    
    #Multiclass classification
    st.header('Multi-class Classification')
    st.subheader('Source: Iris plants dataset')
    st.subheader('Estimator: Random Forest classifier')

    #Load dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, Y)
    iris_predict = clf_rf.predict(df)
    iris_predict_proba = clf_rf.predict_proba(df)

    st.write('Class labels')
    st.write(iris.target_names)

    st.write('Prediction')
    st.write(iris.target_names[iris_predict])

    st.write('Prediction probability')
    st.write(iris_predict_proba)

    if st.checkbox('Show full description of the dataset'):
        st.write(iris.DESCR)
    if st.checkbox('Show dataset table'):
        st.dataframe(pd.DataFrame(iris.data, columns=iris.feature_names))

else:
    st.sidebar.subheader('Input parameters')
    df = diabetes_input()

    st.header('Linear Regression')
    st.subheader('Source: Diabetes dataset')
    st.subheader('Estimator: Linear regressor')

    diabetes = datasets.load_diabetes()
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    
    # Create linear regression object
    regr = LinearRegression()
    
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    # diabetes_y_pred = regr.predict(df)
    
    # The coefficients
    st.write('Coefficients: \n', regr.coef_)
    
    # The mean squared error
    st.write('Mean squared error: %.2f'%mean_squared_error(diabetes_y_test, diabetes_y_pred))
    
    # The coefficient of determination: 1 is perfect prediction
    st.write('Coefficient of determination: %.2f'%r2_score(diabetes_y_test, diabetes_y_pred))
    
    if st.checkbox('Show full description of the dataset'):
        st.write(diabetes.DESCR)
    if st.checkbox('Show dataset table'):
        st.dataframe(pd.DataFrame(diabetes.data, columns=diabetes.feature_names))