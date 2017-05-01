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

# preprocessing stuff
from preprocessingTR import *



if __name__ == '__main__':
	print('Executing testing script...')

	# generate user dictionary
	u_dict = generate_udict('./Data_test/')

	# preprocess and save training data
	preprocess_and_save(u_dict, dir_name='./Data_test/processed/')

	# merge and save files as binary objects for quick loading
	merge_incremental(base='./Data_test/processed/')
