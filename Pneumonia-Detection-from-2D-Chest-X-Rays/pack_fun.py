import numpy as np
import pandas as pd
from random import sample
from numpy.random import seed
seed(333)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

from IPython.display import Image
from skimage import io, color
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import time
import keras.backend as K
from keras.models import Sequential, Model, model_from_json#, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def load_model(model_path, weight_path):
   
    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    
    return model

def get_unique_labels(col):
    """
    This function takes a column, in our case 'Finding Labels'.
    Splits the strings and finds unique label values,
    stores these values in a set.
    """
    labels = set()
    for l in col.unique():
        labels.update(l.split('|'))
#     print('Total unique labels: {}'.format(len(labels)))
    return labels

def split_col_labels(df, labels):
    """
    This function takes a set of labels and a dataframe.
    Splits the labels listed in the column 'Finding Labels'
    into seperate columns with value 1 for found and 0 for absent.
    """
    for l in labels:
        df[l] = df['Finding Labels'].map(lambda x: 1 if l in x else 0)
    return df

def create_splits(df, random_seed):
    df_train, df_val = train_test_split(df,
                                  test_size = 0.2,
                                  random_state= random_seed,
                                  stratify = df['class'])
    
    p_train_ratio = len(df_train[df_train['class'] == 'P'])/len(df_train)
    p_val_ratio = len(df_val[df_val['class'] == 'P'])/len(df_val)
#     print('''Train set: \t\t{}\nVal set: \t\t{}\nTrain % Pneumonia: \t{:.2f}%\nVal % Pneumonia: \t{:.2f}%'''.format(df_train.shape[0],
#                                     df_val.shape[0],
#                                     p_train_ratio*100,
#                                     p_val_ratio*100))
    return df_train, df_val

def balance_train_target_proportions(df_train):
    p_inds = df_train[df_train['class']=='P'].index.tolist()
    np_inds = df_train[df_train['class']=='N'].index.tolist()
    np_sample = sample(np_inds,len(p_inds))
    return df_train.loc[p_inds + np_sample]

def balance_val_target_proportions(df_val):
    p_inds = df_val[df_val['class']=='P'].index.tolist()
    np_inds = df_val[df_val['class']=='N'].index.tolist()
    np_sample = sample(np_inds,len(p_inds)*4) # for a 20/80 split
    return df_val.loc[p_inds + np_sample]


def ig():

    return ImageDataGenerator(
        rescale=1.0/255.0,
        samplewise_center=True,
        samplewise_std_normalization= True,
        horizontal_flip = True, 
        vertical_flip = False, 
        height_shift_range= 0.05, 
        width_shift_range=0.05,
        zoom_range=0.15,
        #zca_whitening=True
    )

def train_image(df_train, img_size):
    
    train_idg = ig()
    train_gen = train_idg.flow_from_dataframe(
        dataframe=df_train,
        shuffle = True,
        x_col = 'img_path',
        y_col = 'class',
        class_mode = 'binary',
        target_size = img_size,
        batch_size = 128
    )
    return train_gen

def val_image(df_val, img_size):
    
    val_idg = ImageDataGenerator(rescale=1. / 255.0) #shouldnt augment ecept for scaling
    val_gen = val_idg.flow_from_dataframe(
        dataframe=df_val,
        shuffle = False,
        x_col = 'img_path',
        y_col = 'class',
        class_mode = 'binary',
        target_size = img_size,
        batch_size = 1024 # use larger batch for eval
    )
    return val_gen

def plot_img_Aug(gen, nrows, ncols):
    t_x, t_y = next(gen)
    fig, m_axs = plt.subplots(nrows, ncols, figsize = (16, 16))
    for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
        c_ax.imshow(c_x[:,:,0], cmap = 'bone')
        if c_y == 1: 
            c_ax.set_title('Pneumonia')
        else:
            c_ax.set_title('No Pneumonia')
        c_ax.axis('off')
        
def load_vgg_model():
    
    model = VGG16(
        include_top=True,
        weights='imagenet'
    )
    transfer_layer = model.get_layer('block5_pool')
    vgg_model = Model(inputs=model.input,
                      outputs=transfer_layer.output)
    
    ## Now, choose which layers of VGG16 we actually want to fine-tune (if any)
    ## Here, we'll freeze all but the last convolutional layer
    for layer in vgg_model.layers[0:17]:
        layer.trainable = False
    
    return vgg_model
    
def plot_history(history, model_name):
    model_name = 'Attention CNN Model'
    N = len(history.history["loss"])
    #plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train loss", color='orange', linestyle='--')
    plt.plot(np.arange(0, N), history.history["val_loss"], label="validation loss", color='purple', linestyle='--')
    plt.plot(np.arange(0, N), history.history["binary_accuracy"], label="train accuracy", color='orange', linestyle='-')
    plt.plot(np.arange(0, N), history.history["val_binary_accuracy"], label="validation accuracy", color='purple', linestyle='-')
    plt.title("Training History")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left");
    plt.savefig('{}-history-plot-tog.png'.format(model_name))
    
def plot_auc(t_y, p_y):
    model_name = 'Attention CNN Model'
    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Pisitive Rate')
    plt.title('ROC Curve')
    plt.savefig('{}-auc-plot.png'.format(model_name))
    plt.show()

def plot_pr(t_y, p_y):
    model_name = 'Attention CNN Model'
    precision, recall, thresholds = precision_recall_curve(t_y, p_y, pos_label=1)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('{}-pr-plot.png'.format(model_name))
    plt.show()
    
def plot_pr_th(t_y, p_y):
    model_name = 'Attention CNN Model'
    precision, recall, thresholds = precision_recall_curve(t_y, p_y, pos_label=1)
    plt.plot(thresholds, precision[:-1], color='red', lw=2, label='precision')
    plt.plot(thresholds, recall[:-1], color='blue', lw=2, label='recall')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Precision-Recall by Threshold')
    plt.savefig('{}-prth-plot.png'.format(model_name))
    plt.show()
    
def calc_f1(prec,recall):
    return 2*(prec*recall)/(prec+recall) if recall and prec else 0

def find_optimal_th(y_val, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    f1score = [calc_f1(precision[i],recall[i]) for i in range(len(thresholds))]
    idx = np.argmax(f1score)
    
    plt.figure()
    plt.plot(thresholds, f1score)
    plt.title("F1 score vs. threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.show()

    print('Precision: '+ str(precision[idx]))
    print('Recall: '+ str(recall[idx]))
    print('Threshold: '+ str(thresholds[idx]))
    print('F1 Score: ' + str(f1score[idx]))
    
    return thresholds[idx]