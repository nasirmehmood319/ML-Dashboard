import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,confusion_matrix
import time
import os
import annotated_text as ant 
from annotated_text import annotation 









st.set_page_config(page_title="Sharp.X", page_icon=":bar_chart:", layout="wide")

st.title("Machine Learning")
ant.annotated_text(annotation("BETA 1.0", border='3px groove #00FFEF')) 

st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)


st.write('Your dataset must be cleaned:')
uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file,encoding='ISO-8859-1')
    st.write(dataframe)
else:
    print("File Not Selected")
try:
    
    col1, col2 = st.columns((2))
    with col1:
        selected_fea =  st.multiselect("Pick your features(values of X):", dataframe.columns)
        X = dataframe[selected_fea]
        st.write(X)
    with col2:
        selected_lab = st.multiselect("Pick your Label(values of y):", dataframe.columns)
        y = dataframe[selected_lab]
        st.write(y)
    def split_data():
        global X_train,X_test,y_train,y_test 
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)
            

    st.write("Shape of your dataset for Training and Testing")

    st.button("Splited Dataset",on_click=split_data())
    st.write("X_train shape: ",X_train.shape)
    st.write("y_train shape: ",y_train.shape)
    st.write("X_test shape: ",X_test.shape)
    st.write("y_test shape: ",y_test.shape)

    col1, col2 = st.columns((2))
    with col1:
        options = st.selectbox(
        'Select your method:',
        ('None','Regression', 'Classification'))


    with col2:


        if options == 'Regression':
            estimators = all_estimators(type_filter='regressor') 
            all_models_name = []
            all_models = []
            for name, get_model in estimators:
                all_models_name.append(name)
                all_models.append(get_model)
            all_models_name = tuple(all_models_name)
            regression_options = st.selectbox(
            'Select your model:',
            (all_models_name))
            # st.write("you select ",regression_options)
            for name, get_model in estimators:
                if name == regression_options:
                    model = get_model()
                    model.fit(X_train,y_train)
                    st.write(model)

            score = st.selectbox(
            'Select your score method:',
            ("R2 Score","Mean Squared Error","Mean Absolute Error","Mean Absolute Percentage Error"))
            if score == "R2 Score":
                estimators = all_estimators(type_filter='regressor') #classifier
                for name, get_model in estimators:
                    if name == regression_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = r2_score(y_test,pred_y)
                        st.write(acc)
            elif score == "Mean Squared Error":
                estimators = all_estimators(type_filter='regressor') #classifier
                for name, get_model in estimators:
                    if name == regression_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = mean_squared_error(y_test,pred_y)
                        st.write(acc)
            elif score == "Mean Absolute Error":
                estimators = all_estimators(type_filter='regressor') #classifier
                for name, get_model in estimators:
                    if name == regression_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = mean_absolute_error(y_test,pred_y)
                        st.write(acc)
            elif score == "Mean Absolute Percentage Error":
                estimators = all_estimators(type_filter='regressor') #classifier
                for name, get_model in estimators:
                    if name == regression_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = mean_absolute_percentage_error(y_test,pred_y)
                        st.write(acc)

        elif options == 'Classification':
            estimators = all_estimators(type_filter='classifier') #classifier
            all_models_name = []
            for name, get_model in estimators:
                all_models_name.append(name)
            all_models_name = tuple(all_models_name)
            classification_options = st.selectbox(
            'Select your model:',
            (all_models_name))
            for name, get_model in estimators:
                if name == classification_options:
                    model = get_model()
                    model.fit(X_train,y_train)
                    st.write(model)

            score = st.selectbox(
                'Select your score method',
                ("Accuracy Score","Precision Score","F1 Score","Recall Score"))
            if score == "Accuracy Score":
                estimators = all_estimators(type_filter='classifier') #classifier
                for name, get_model in estimators:
                    if name == classification_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = accuracy_score(y_test,pred_y)
                        cm = confusion_matrix(y_test,pred_y)
                        st.write(acc)
                        st.write("Confusion Matrix:")
                        st.write(cm)

            elif score == "Precision Score":
                estimators = all_estimators(type_filter='classifier') #classifier
                for name, get_model in estimators:
                    if name == classification_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = precision_score(y_test,pred_y,average='weighted')
                        st.write(acc)
                        cm = confusion_matrix(y_test,pred_y)
                        st.write("Confusion Matrix:")
                        st.write(cm)
            elif score == "F1 Score":
                estimators = all_estimators(type_filter='classifier') #classifier
                for name, get_model in estimators:
                    if name == classification_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = f1_score(y_test,pred_y,average='weighted')
                        st.write(acc)
                        cm = confusion_matrix(y_test,pred_y)
                        st.write("Confusion Matrix:")
                        st.write(cm)
            elif score == "Recall Score":
                estimators = all_estimators(type_filter='classifier') #classifier
                for name, get_model in estimators:
                    if name == classification_options:
                        model = get_model()
                        model.fit(X_train,y_train)
                        pred_y = model.predict(X_test)
                        acc = recall_score(y_test,pred_y,average='weighted')
                        st.write(acc)
                        cm = confusion_matrix(y_test,pred_y)
                        st.write("Confusion Matrix:")
                        st.write(cm)
except:
    st.write("File Not Selected")
