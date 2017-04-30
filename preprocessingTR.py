import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, re, sys

# sklearn stuff
from sklearn.decomposition import PCA

# scipy stuff
from scipy.interpolate import interp1d

def delete_relevant(folder):
    if folder[-1] != '/':
        folder += '/'
    files = os.listdir(folder)
    for f in files:
        if re.match(r'u[0-9]+_w[0-9]+_(accelerometer|gyroscope)\.log', f) == None:
            print('deleting %s' % f)
            os.remove(folder + f)

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

def filter_datapoints(data, threshold=0.25):
    """
    input:
        data (list of lists containing individual cycle values in each list)
        threshold [optional] (threshold value for length of lists to include
        eg. 0.25 mean lists of 0.75 < len < 1.25)
    output: list of lists
    
    filters out very long lists and very short lists
    """
    # immediately remove samples less than 80 or greater than 300
    lengths = [len(x) for x in data if len(x) >= 80 and len(x) <= 300]
    mean_len = np.mean(lengths)
    return [x for x in data if len(x) <= (1 + threshold) * mean_len and len(x) > (1 - threshold) * mean_len]

def extract_feats(data, starts, threshold=0.25, filter_short=True):
    """
    input:
        data (time series)
        starts (start of each cycle)
        filter_short [optional] (filters out short cycles)
    output: out (list of lists containing individual cycle values in each list)
    """
    ns = len(starts)
    out = []
    for i in range(ns-1):
        out.append(data[starts[i]:starts[i+1]])
    if filter_short:
        return filter_datapoints(out, threshold)
    return out

def plot_steps(data, starts, main_title='Accelerometer Magnitude', main_ylabel='$m/s^2$'):
    """
    plots overall time series with starts overlayed
    plots individual walking cycles
    """
    
    plt.figure(figsize=(15,4))
    plt.plot(data)
    plt.title(main_title)
    plt.xlabel('samples')
    plt.ylabel(main_ylabel)
    mx = data.max()
    mn = data.min()
    vy = np.linspace(mn, mx, 2)

    for s in starts:
        vx = [s for _ in vy]
        plt.plot(vx, vy, c='r', linewidth=2)
    plt.show()
    
    plt.figure(figsize=(15,4))
    plt.title('Individual Walking Cycles')
    plt.xlabel('samples')
    plt.ylabel(main_ylabel)
    for s in extract_feats(data, starts):
        plt.plot(s)

def interpolate_features(feats, num=300):
    """
    input:
        feats (list of lists containing individual cycle values in each list)
        num [optional] (number of points to interpolate to)
    output: feats_interp (interpolated list of lists of same length)
    """
    feats_interp = []
    for point in feats:
        n = len(point)
        f = interp1d(np.arange(n), point, kind='cubic')
        point_interp = np.linspace(0, n-1, num=num, endpoint=True)
        feats_interp.append(f(point_interp))
    feats_interp = np.array(feats_interp)
    return feats_interp

def rot_trans(raw_data):
    """
    input: raw_data
    output: rotation invariant data via linear PCA
    """
    n, d = raw_data.shape
    pca = PCA(n_components=d)
    return pca.fit_transform(raw_data)

def merge_consecutive_starts(starts, num_merges=2):
    return starts[::num_merges]

def preprocess_data(df, begin_idx=1000, threshold=0.25, num_merges=1):
    """
    input: df (dataframe of accelerometer readings. 1 column for each axis. Each row must be at one timestamp)
    output: out (dataframe of interpolated, distinct step data values)
    """
    # keep accelerometer columns
    accelerometer_cols = [x for x in df.columns if 'data' in x]
    # rotation invariance transformation
    data = rot_trans(df[accelerometer_cols].values)
    # toss first few values to noise/errors
    data = data[begin_idx:, :]
    
    # calculate accelerometer magnitude
    acc_mag = (data ** 2).sum(axis=1) ** 0.5
    
    # calculate start of each cycle
    starts = merge_consecutive_starts(find_cycles(acc_mag), num_merges=num_merges)
    
    # [n x k*c] matrix where
    # n = number of distinct cycles
    # k = number of axes in accelerometer
    # c = number of interpolated columns
    out = []
    columns = []
    for idx, row in enumerate(data.T):
        feats = extract_feats(row, starts, threshold=threshold, filter_short=True)
        feats_interp = interpolate_features(feats)
        n, d = feats_interp.shape
        out.append(feats_interp)
        ctr = 0
        for i in range(d):
            columns.append('axis_%d_feat_%d' % (idx, i))
    out = np.concatenate(out, axis=1)
    return pd.DataFrame(out, columns=columns)

def preprocess_and_save(u_dict, dir_name='./Data/processed/'):
    s = 0
    for key in u_dict:
        s += len(u_dict[key])

    ctr = 0
    for uid in u_dict:
        xfname = dir_name + '%d_X.csv' % (uid)
        yfname = dir_name + '%d_Y.npy' % (uid)
        
        # check if files exist
        if not (os.path.isfile(xfname) and os.path.isfile(yfname)):
            # get trials for user
            trials = u_dict[uid]
            data = []
            labels = []
            # for each trial
            for trial in trials:
                ctr += 1
                print('Processing user %d, trial %d, %d remaining' % (uid, trial, (s-ctr)))
                try:
                    # get data from disk
                    df = get_data(uid, trial)
                    # proprocess data, also choose number of strides to combine
                    d = preprocess_data(df, num_merges=2)
                    # append to combined data
                    data.append(d)
                    # append to combined labels
                    n,_ = d.shape
                    labels.append(np.zeros(n) + uid)
                except:
                    print('FAILED')

            # save data and labels
            if len(data) != 0:
                data = pd.concat(data, axis=0, ignore_index=True)
                labels = np.concatenate(labels)
                data.to_csv(xfname)
                np.save(yfname, labels)
        else:
            print('Files for user %d exist on disk.' % uid)

def merge_and_save(base='./Data/processed/'):
    builder = pd.DataFrame()
    labels = []
    for f in os.listdir(base):
        if re.match(r'[0-9]+_X.csv', f):
            uid = int(re.findall(r'([0-9]+)_X.csv', f)[0])
            print('Loading user %d' % uid)
            data = pd.read_csv(base + f, index_col=0)
            n, _ = data.shape
            for i in range(n):
                labels.append(uid)
            if builder.shape == (0,0):
                builder = data
            else:
                builder = pd.concat([builder, data], axis=0)

    builder.to_pickle(base + 'full.pickle')
    np.save(base+'labels.npy', labels)


if __name__ == "__main__":
    print('Executing preprocessing script...\n')

    # generate user dictionary
    u_dict = generate_udict('./Data/')

    # preprocess and save training data
    preprocess_and_save(u_dict, dir_name='./Data/processed/')

    # merge and save files as binary objects for quick loading
    merge_and_save(base='./Data/processed/')