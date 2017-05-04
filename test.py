import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, re, sys, time

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
    ss = joblib.load(model_dir + 'ss.pkl')
    xTe = ss.transform(xTe)

    return xTe

if __name__ == '__main__':
    print('Executing testing script...')
    
    print('Loading model')
    t0 = time.time()
    model = keras.models.load_model('./Models/ConvNetC.h5')
    t1 = time.time()
    print('Loaded model in %.2fs' % (t1 - t0))
    
    user_lookup = {0: 'other', 1: 'Mukund', 2: 'Frank'}
    user2netid = {0: 'other', 1:'ms2666', 2: 'fc249'}
    
    while True:
        if not os.path.isfile('./Data_test/u000_w000/u000_w000_accelerometer.log'):
            print('No new data detected')
        else:
            print('Running network')

            with open('./Data_test/RUNNING', 'wb') as f:
                pass
            
            try:
                ## << code in
                t0 = time.time()
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

                preds = model.predict(xTe_conv)

                id_predicted = mode(preds.round().argmax(axis=1)).mode[0]

                t1 = time.time()
                
                fig, ax = plt.subplots()
                fig.set_figheight(8)
                fig.set_figwidth(3)
                ax.bar([0, 1, 2], preds.mean(axis=0))
                ax.set_xticks([0,1,2])
                ax.set_xticklabels(['Other', 'Mukund', 'Frank'])
                ax.title.set_text('Softmax Probabilities')
                fig.savefig('./Data_test/results.png')



                print('Finished testing in %.2fs' % (t1-t0))
                ## >> code out
            except:
                id_predicted = 0
                
            if id_predicted == 0:
                print('Get the fuck out of here!')
            else:
                print('Welcome back %s!' % user_lookup[id_predicted])
            # save output
            with open('./Data_test/RESULT', 'w') as f:
                f.write('%s' % user2netid[id_predicted])
            
            
            # delete cache files and u000_w000_accelerometer.log
            cache_dir = './Data_test/processed/'
            for cachefile in os.listdir(cache_dir):
                os.remove(cache_dir + cachefile)
            os.remove('./Data_test/u000_w000/u000_w000_accelerometer.log')
            os.remove('./Data_test/RUNNING')

        time.sleep(1)

