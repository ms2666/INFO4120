import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, re, sys

# sklearn stuff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

# keras stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras import callbacks

# scipy stuff
from scipy.interpolate import interp1d
from scipy.stats import mode

# preprocessing stuff
from preprocessingTR import *

def load_data(xpath='./Data_test/processed/full.pickle'):
    return pd.read_pickle(xpath)

def scale_data(df, model_dir='./Models/'):
    """
    Scale data
    """
    xTe = df.values
    num_classes = int(labels.max())
    ss = joblib.load(model_dir + 'ss.pkl')
    xTe = ss.transform(xTe)

    return xTe

if __name__ == '__main__':
    print('Executing testing script...')

    # generate user dictionary
    u_dict = generate_udict('./Data_test/')

    # preprocess and save training data
    preprocess_and_save(u_dict, dir_name='./Data_test/processed/')

    # merge and save files as binary objects for quick loading
    merge_incremental(base='./Data_test/processed/')

    # load and split data
    data = load_data()
    xTe = scale_data(data)
    
    print('Reshaping data')
    xTe_conv = xTe.reshape(-1, 3, 300, 1)
    
    print('Loading model')
    model = keras.models.load_model('./Models/ConvNetC.h5')
    
    preds = model.predict(xTe_conv)
    
    user_lookup = {0: 'other', 1: 'Mukund', 2: 'Frank'}
    
    id_predicted = mode(preds.round().argmax(axis=1)).mode[0]
    
    if id_predicted == 0:
        print('Get the fuck out of here!')
    else:
        print('Welcome back %s!' % user_lookup[id_predicted])