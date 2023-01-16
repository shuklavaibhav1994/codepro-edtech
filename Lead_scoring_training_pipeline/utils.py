###############################################################################
# Import necessary modules
# Developer: Vaibhav Shukla
# ##############################################################################

#importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlite3 import Error
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import lightgbm as lgb
from datetime import date
from Lead_scoring_training_pipeline.constants import *


###############################################################################
# Define the function to encode features
# ##############################################################################

def encode_features():
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        DB_PATH : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''
    # read the model input data
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_model_input = pd.read_sql('select * from interactions_mapped', cnx)

    # create df to hold encoded data and intermediate data
    df_encoded = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
    df_placeholder = pd.DataFrame()

    for f in FEATURES_TO_ENCODE:
        if(f in df_model_input.columns):
            encoded = pd.get_dummies(df_model_input[f])
            encoded = encoded.add_prefix(f + '_')
            df_placeholder = pd.concat([df_placeholder, encoded], axis=1)
        else:
            print('Feature not found')
            return df_model_input

    # add the encoded features into a single dataframe
    for feature in df_encoded.columns:
        if feature in df_model_input.columns:
            df_encoded[feature] = df_model_input[feature]
        if feature in df_placeholder.columns:
            df_encoded[feature] = df_placeholder[feature]
    df_encoded.fillna(0, inplace=True)

    print("Encoded dataframe columns: ", df_encoded.columns)
    
    df_encoded.to_sql(name='features', con=cnx,
                       if_exists='replace', index=False)
    # save the features and target in separate tables
    df_target = df_model_input[['app_complete_flag']]
    
    df_target.to_sql(name='target', con=cnx, if_exists='replace', index=False)

    cnx.close()

###############################################################################
# Define the function to create mlflow experiments
# ##############################################################################

def create_mlflow_experiment():
    #experiment_name = EXPERIMENT+'_'+date.today().strftime("%d_%m_%Y")
    experiment_name = EXPERIMENT
    
    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        # Creating an experiment
        print("Creating mlflow experiment with name: ", experiment_name)
        mlflow.create_experiment(experiment_name)
    except:
        pass

    # Setting the environment with the created experiment
    print("Setting mlflow experiment to name: ", experiment_name)
    mlflow.set_experiment(experiment_name)
    
###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''

    print("Set MLflow tracking url and create/set experiment")
    create_mlflow_experiment()
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    X = pd.read_sql('select * from features', cnx)
    y = pd.read_sql('select * from target', cnx)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)
    
    mlflow_run_name = ML_FLOW_RUN_NAME+date.today().strftime("%d_%m_%Y_%H_%M_%S")
    with mlflow.start_run(run_name=mlflow_run_name, nested=True) as run:
        clf = lgb.LGBMClassifier()
        clf.set_params(**model_config)
        clf.fit(X_train, y_train)
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="models", registered_model_name='LightGBM')
        mlflow.log_params(model_config)
        
        # predict the results on training dataset
        print("Making prediction on test data")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        conf_mat = confusion_matrix(y_pred, y_test)
        precision = precision_score(y_pred, y_test,average= 'macro')
        recall = recall_score(y_pred, y_test, average= 'macro')
        f1 = f1_score(y_pred, y_test, average='macro')
        try:
            roc_auc = roc_auc_score(y_pred, y_test)
        except ValueError:
            pass
        cm = confusion_matrix(y_test, y_pred)
        
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
        class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)
        print("Acc: " + str(acc) + ", Precision: " + str(precision) + ", Recall: " + str(recall) + ", f1: " + str(f1) + ", AUC: " + str(roc_auc))

        print("Logging metrics in MLflow")
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("Precision_0", class_zero[0])
        mlflow.log_metric("Precision_1", class_one[0])
        mlflow.log_metric("Recall_0", class_zero[1])
        mlflow.log_metric("Recall_1", class_one[1])
        mlflow.log_metric("f1_0", class_zero[2])
        mlflow.log_metric("f1_1", class_one[2])
        mlflow.log_metric("False Negative", fn)
        mlflow.log_metric("True Negative", tn)

        runID = run.info.run_uuid
        print("Inside MLflow Run with id {}".format(runID))
        print("Finished creation and registration of trained model in MLflow")
        
    print("Closing database connection")
    cnx.close()


