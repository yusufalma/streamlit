import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier

st.title('ALMA: A Lesson to Machine_learning and Artificial_intelligence')
# st.sidebar.header('Machine Learning Application')

#Load classification dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Load regression dataset
diabetes = datasets.load_diabetes()
diabetes_un = datasets.load_diabetes(scaled=False)
#diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes.data
diabetes_y = diabetes.target
diabetes_X_un = diabetes_un.data
diabetes_y_un = diabetes_un.target

# Use only one feature
#diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

def iris_input():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

def diabetes_input():
    age = (st.sidebar.slider('Age', 19, 79, 48)-diabetes_X_un[:,0].mean())/(diabetes_X_un[:,0].std()*21)
    sex = (st.sidebar.slider('Sex', 1, 2, 1)-diabetes_X_un[:,1].mean())/(diabetes_X_un[:,1].std()*21)
    bmi = (st.sidebar.slider('Body Mass Index', 18.0, 42.2, 26.4)-diabetes_X_un[:,2].mean())/(diabetes_X_un[:,2].std()*21)
    bp = (st.sidebar.slider('Average blood pressure', 62, 133, 95)-diabetes_X_un[:,3].mean())/(diabetes_X_un[:,3].std()*21)
    s1 = (st.sidebar.slider('T-Cells \(a type of white blood cells\)', 97, 301, 189)-diabetes_X_un[:,4].mean())/(diabetes_X_un[:,4].std()*21)
    s2 = (st.sidebar.slider('Low-Density Lipoproteins \(LDL\)', 41.6, 242.4, 115.4)-diabetes_X_un[:,5].mean())/(diabetes_X_un[:,5].std()*21)
    s3 = (st.sidebar.slider('High-Density Lipoproteins \(HDL\)', 22, 99, 50)-diabetes_X_un[:,6].mean())/(diabetes_X_un[:,6].std()*21)
    s4 = (st.sidebar.slider('Thyroid Stimulating Hormone', 2.00, 9.09, 4.07)-diabetes_X_un[:,7].mean())/(diabetes_X_un[:,7].std()*21)
    s5 = (st.sidebar.slider('Lamotrigine', 3.26, 6.11, 4.64)-diabetes_X_un[:,8].mean())/(diabetes_X_un[:,8].std()*21)
    s6 = (st.sidebar.slider('Blood sugar level', 58, 124, 91)-diabetes_X_un[:,9].mean())/(diabetes_X_un[:,9].std()*21)
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
    
    # Create linear regression object
    regr = LinearRegression()
    
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    
    # Make predictions using the testing set
    diabetes_y_val = regr.predict(diabetes_X_test)
        
    # The coefficients
    st.write('Coefficients: \n', regr.coef_)
    
    # The mean squared error
    st.write('Mean squared error: %.2f'%mean_squared_error(diabetes_y_test, diabetes_y_val))
    #st.write('Mean squared error of prediction input: %.2f'%mean_squared_error(diabetes_y_test, diabetes_y_pred))
    
    # The coefficient of determination: 1 is perfect prediction
    st.write('Coefficient of determination: %.2f'%r2_score(diabetes_y_test, diabetes_y_val))
    #st.write('Coefficient of determination of prediction input: %.2f'%r2_score(diabetes_y_test, diabetes_y_pred))
    st.write('Measure of disease progression one year after baseline based on prediction input:', regr.predict(df))
    
    if st.checkbox('Show full description of the dataset'):
        st.write(diabetes.DESCR)
    if st.checkbox('Show dataset table'):
        st.dataframe(pd.DataFrame(diabetes_un.data, columns=diabetes_un.feature_names))