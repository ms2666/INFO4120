import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os, re, sys

# sklearn stuff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
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
    num_classes = int(labels.max())
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
    n = 2*(labels == target_class).sum()
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

def trinarize(labels, class1, class2):
    out = []
    for l in labels:
        if l == class1:
            out.append(1)
        elif l == class2:
            out.append(2)
        else:
            out.append(0)
    return np.array(out)

if __name__ == '__main__':
    print('Executing training script...')

    print('Loading data')
    data, labels = load_data()
    xTr, xTe, yTr, yTe = split_data(data, labels, model_dir='./Models/')

    print('Reshaping data')
    xTr_conv = xTr.reshape(-1, 3, 300, 1)
    xTe_conv = xTe.reshape(-1, 3, 300, 1)

    muk_idx_tr = random_others(yTr, 51)
    muk_idx_te = random_others(yTe, 51)

    fra_idx_tr = random_others(yTr, 0)
    fra_idx_te = random_others(yTe, 0)

    tr_idx = np.logical_or(muk_idx_tr, fra_idx_tr)
    te_idx = np.logical_or(muk_idx_te, fra_idx_te)

    print('Converting labels to categorical')
    yTr_tri = trinarize(yTr[tr_idx], 51, 0)
    yTe_tri = trinarize(yTe[te_idx], 51, 0)

    yTr_cat = keras.utils.to_categorical(yTr_tri, num_classes=3)
    yTe_cat = keras.utils.to_categorical(yTe_tri, num_classes=3)

    print('Compiling model')
    model = Sequential([
        Conv2D(128, (3,50), activation='relu', input_shape=(3,300,1), padding='same'),
        Conv2D(128, (3,30), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3,10)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    def f1score(y_true, y_pred):
        return f1_score(y_true, y_pred, 'weighted')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print('Training model')
    history = model.fit(xTr_conv[tr_idx,:,:,:],
                    yTr_cat,
                    epochs=30,
                    batch_size=128,
                    validation_data=(xTe_conv[te_idx,:,:,:], yTe_cat))

    preds = model.predict(xTe_conv[te_idx, :, :, :])

    fig, axarr = plt.subplots(2, 1, sharex=True)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    
    axarr[0].plot(history.history['loss'])
    axarr[0].set_ylabel('Loss')
    
    axarr[1].plot(history.history['acc'])
    axarr[1].set_xlabel('Epoch')
    axarr[1].set_ylabel('Accuracy')

    fig.savefig('./Data/train_results.png')

    conf = confusion_matrix(yTe_cat.argmax(axis=1), preds.argmax(axis=1))

    fig, ax = plt.subplots()
    fig.suptitle('Confusion Matrix', fontsize = 20)
    fig.subplots_adjust(top=0.8)
    ax.matshow(conf, cmap=plt.cm.Blues)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')

    for (i, j), z in np.ndenumerate(conf):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    fig.savefig('./Data/confmat.png')

    print('ACCURACY: %.4f' % (yTe_cat == preds.round()).mean())
    print('F1: %.4f' % f1_score(yTe_cat.argmax(axis=1), preds.argmax(axis=1), average='weighted'))
    print('Confusion Matrix:')
    print(confusion_matrix(yTe_cat.argmax(axis=1), preds.argmax(axis=1)))
    
    print('Saving model')
    model.save('./Models/ConvNetC.h5')
    
