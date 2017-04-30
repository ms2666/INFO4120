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


def load_data(xpath='./Data/processed/full.pickle', ypath='./Data/processed/labels.npy'):
	return pd.read_pickle(xpath), np.load(ypath)


def split_data(df, labels, model_dir='./Models/', test_size=0.2):
	"""
	Split and scale data, convert to 0 indexing
	"""
	xTr, xTe, yTr, yTe = train_test_split(df.values, labels, test_size=test_size)
	num_classes = labels.max()
	ss = StandardScaler()
	xTr = ss.fit_transform(xTr)
	xTe = ss.transform(xTe)

	# save standard scaler model
	joblib.dump(ss, model_dir + 'ss.pkl')

	# zero index labels
	yTr = np.mod(yTr, num_classes)
	yTe = np.mod(yTe, num_classes)

	return xTr, xTe, yTr, yTe

def random_others(labels, target_class):
    out = []
    ctr = 0
    n = (labels == target_class).sum()
    for i in range(len(labels)):
        if labels[i] == target_class:
            out.append(True)
        else:
            p = 0.5
            if np.random.rand() > p and ctr < n:
                ctr += 1
                out.append(True)
            else:
                out.append(False)
    return np.array(out)

### Convert labels to categorical

if __name__ == '__main__':
	print('Executing training script...')

	remote = callbacks.RemoteMonitor(root='http://localhost:9000')

	data, labels = load_data()
	xTr, xTe, yTr, yTe = split_data(data, labels)

	xTr_conv = xTr.reshape(-1, 3, 300, 1)
	xTe_conv = xTe.reshape(-1, 3, 300, 1)

