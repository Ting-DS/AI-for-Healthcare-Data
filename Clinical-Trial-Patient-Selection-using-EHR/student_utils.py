# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae

# Aequitas
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.bias import Bias
from aequitas.fairness import Fairness

import functools

import matplotlib.pyplot as plt
import seaborn as sns
    
from PIL import Image
import IPython.display as display

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    train, val_test = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        shuffle=True,
        stratify=df[['time_in_hospital']])

    validation, test = train_test_split(
        val_test,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=val_test[['time_in_hospital']])

    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        
        num_lines = sum(1 for _ in open(vocab_file_path))
        cat_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file=vocab_file_path, num_oov_buckets=0)       
        
        if num_lines > 10:
            dims = 10
            print(f'\n{c} \t- # lines: {num_lines}, embedding (categorical)')
            tf_categorical_feature_column = tf.feature_column.embedding_column(cat_column, dimension=dims)
        else:
            print(f'\n{c} \t- # lines: {num_lines}, indicator (categorical)')
            tf_categorical_feature_column = tf.feature_column.indicator_column(cat_column)
        
        output_tf_list.append(tf_categorical_feature_column)
        
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    print(f'\n{col} \t- mean: {MEAN} - std: {STD}, numeric (normalized)')
    
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col,
        default_value=default_value,
        normalizer_fn=normalizer,
        dtype=tf.float64)

    return tf_numeric_feature

def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >= 5 else 0).values
    print(student_binary_prediction.shape)
    print(type(student_binary_prediction))
    return student_binary_prediction




#Other

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns    
    
def add_pred_to_test(test_df, pred_np, demo_col_list):
    for c in demo_col_list:
        test_df[c] = test_df[c].astype(str)
    test_df['score'] = pred_np
    test_df['label_value'] = test_df['time_in_hospital'].apply(lambda x: 1 if x >=5 else 0)
    return test_df

def get_metrics(y_val,y_pred):
    f1 = f1_score(y_val, y_pred, average='weighted')
    class_report = classification_report(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='micro')
    recall = recall_score(y_val, y_pred, average='micro')
    return f1, class_report, auc, precision, recall

def plot_auc(y_val, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_val, y_pred, pos_label=1)
    plt.plot(fpr, tpr, color='red', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.savefig('{}-auc-plot.png'.format(model_name))
    plt.show()

def plot_pr(y_val, y_pred, model_name):
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred, pos_label=1)
    plt.plot(recall, precision, color='red', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('{}-pr-plot.png'.format(model_name))
    plt.show()