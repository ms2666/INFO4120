import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, re, sys

# scipy stuff
from scipy.interpolate import interp1d

base = './Data/'

def generate_udict(base):
	u_dict = {}
	files = [x for x in os.listdir(base) if os.path.isdir(base+x) and 'u' in x]
	for f in files:
	    uid, trial = re.findall(r'u([0-9]+)_w([0-9]+)', f)[0]
	    uid, trial = int(uid), int(trial)
	    if uid not in u_dict:
	        u_dict[uid] = [trial]
	    else:
	        u_dict[uid].append(trial)
	return u_dict

def gen_fname(uid, trial):
    f = lambda x: ('%3d' % x).replace(' ', '0')
    return 'u%s_w%s' % (f(uid), f(trial))

def get_data(uid, trial):
    fname = gen_fname(uid, trial)
    d1 = pd.read_csv('./Data/%s/%s_accelerometer.log' % (fname, fname), delimiter='\t')
    return d1

def corr(x1, x2):
    """
    input: x1, x2 (vectors of equal length)
    output: pearson correlation coefficient
    """
    n = len(x1)
    return (x1.dot(x2) - n * x1.mean() * x2.mean())/(n * x1.std() * x2.std())

def find_cycles(data, step=150, slack=20, threshold=0.45, viz=False):
    """
    input: data [accelerometer magnitude time series]
    output: starts [start of each step]
    
    find starts of each cycle
    default vales for:
        step = 150
        slack = 20
        threshold = 0.45
    """
    counter = 0
    
    N = len(data)
    start_idx = np.argmin(data[:step])
    end_idx = start_idx+step-slack + np.argmin(data[start_idx+step-slack:start_idx+step+slack])

    R1 = data[np.arange(start_idx, end_idx)]
    L = end_idx - start_idx
    
    c = []
    for i in range(N-L):
        Di = data[i:i+L]
        ci = corr(R1, Di)
        c.append(ci)
    
    if viz:
        plt.figure(figsize=(15,4))
        plt.title('Correlation')
        plt.plot(c)
    
    starts = []
    for i in range(1, N-L-1):
        if c[i] > c[i-1] and c[i] > c[i+1] and c[i] > threshold:
            starts.append(i)
    
    return starts

