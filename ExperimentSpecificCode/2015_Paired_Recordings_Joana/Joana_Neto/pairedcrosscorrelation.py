

import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import itertools
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.signal as signal



def crosscorrelate(sua1, sua2, lag=None, n_pred=1, predictor=None,
                   display=False, kwargs={}):

    assert predictor is 'shuffle' or predictor is None, "predictor must be \
    either None or 'shuffle'. Other predictors are not yet implemented."
    #Check whether sua1 and sua2 are SpikeTrains or arrays
    sua = []
    for x in (sua1, sua2):
        #if isinstance(x, SpikeTrain):
        if hasattr(x, 'spike_times'):
            sua.append(x.spike_times)
        elif x.ndim == 1:
            sua.append(x)
        elif x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
            sua.append(x.ravel())
        else:
            raise TypeError("sua1 and sua2 must be either instances of the" \
                            "SpikeTrain class or column/row vectors")
    sua1 = sua[0]
    sua2 = sua[1]
    if sua1.size < sua2.size:
        if lag is None:
            lag = np.ceil(10*np.mean(np.diff(sua1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20*np.mean(np.diff(sua2)))
        sua1, sua2 = sua2, sua1
        reverse = True
    #construct predictor
    if predictor is 'shuffle':
        isi = np.diff(sua2)
        sua2_ = np.array([])
        for ni in range(1, n_pred+1):
            idx = np.random.permutation(isi.size-1)
            sua2_ = np.append(sua2_, np.add(np.insert(
                (np.cumsum(isi[idx])), 0, 0), sua2.min() + (
                np.random.exponential(isi.mean()))))
    #calculate cross differences in spike times
    differences = np.array([])
    pred = np.array([])
    for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
        differences = np.append(differences, sua1[k] - sua2[np.nonzero(
            (sua2 > sua1[k] - lag) & (sua2 < sua1[k] + lag))])
    if predictor == 'shuffle':
        for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
            pred = np.append(pred, sua1[k] - sua2_[np.nonzero(
                (sua2_ > sua1[k] - lag) & (sua2_ < sua1[k] + lag))])
    if reverse is True:
        differences = -differences
        pred = -pred
    norm = np.sqrt(sua1.size * sua2.size)
    return differences, pred, norm



############## 32 channels ##########################

def open_hdf5_kwx(filename):

    f = h5py.File(filename,'r')
    dataset = f['/channel_groups/0/features_masks']
    #dataset = f['/channel_groups/1/features_masks']
    dataset.shape
    teste= dataset[:,:,1]
    gh=teste[:,slice(0,96,3)]
    return np.array(gh)



def open_hdf5_kw(filename):
    f = h5py.File(filename,'r')
    times = f['/channel_groups/0/spikes/time_samples']
    #times = f['/channel_groups/1/spikes/time_samples']
    return np.array(times)


def times_spikes_ch(maskdata,timedata):
    global time_ch

    time_ch=[]
    for i in np.arange(32):
        maskdata_ch=maskdata[:,i]
        t=[timedata[x] for x in range(np.shape(timedata)[0]) if maskdata_ch[x]>0]
        time_ch.append(t)
        i=i+1


# How to plot the histograms in space for 32channels?

cellname= '2014_0326_25_pair2.0'
maskdata= open_hdf5_kwx(r'E:\Data\2014-03-26\Klusta_spikes\amplifier_depth_90um_s16.kwx')
timedata= open_hdf5_kw(r'E:\Data\2014-03-26\Klusta_spikes\amplifier_depth_90um_s16.kwik')
times_spikes_ch(maskdata,timedata)#return time_ch

juxta_times= all_cells_spike_triggers['2']

def plot_crosscorrelation(juxta,timematriz, ysup=40, x=50):
    global cross
    cross = []
    juxta=np.asarray(juxta, dtype=np.float64)
    for i in np.arange(32):
        channel= (np.array(timematriz[i])).astype(np.float64)
        cross_array= crosscorrelate(channel, juxta, lag=1500)
        cross.append(cross_array)

    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    for i in np.arange(32):
        plt.subplot(22,3,subplot_order [i])
        plt.hist((cross[sites_order_geometry[i]][0])/30,bins=101, range=(-50,50), align='mid')
        plt.ylim(0, ysup)
        plt.xlim(-x,x)


plot_crosscorrelation(juxta_times, time_ch)

# MAX of detections for each channel
max=[]
index=[]
plt.figure()
for i in np.arange(32):
    n,b,p=plt.hist((cross[i][0])/30,bins=101, range=(-50,50), align='mid')
    max.append(n.max())
    index.append(n.argmax())

max = np.array(max)
index = np.array(index)

print(max.max())
print(max.argmax())


####cross w times from spikes

def plot_crosscorrelation(juxta,timedata,ysup=10000, x=50):
    global cross
    cross = []

    juxta=np.asarray(juxta, dtype=np.float64)
    channel= (np.array(timedata)).astype(np.float64)
    cross = crosscorrelate(channel, juxta, lag=1500)
    plt.figure()
    n,b,p=plt.hist((cross[0])/30,bins=101, range=(-50,50), align='mid')
    max = n.max()
    index= n.argmax()
    print(max.max())
    print(max.argmax())
    plt.ylim(0, ysup)
    plt.xlim(-x,x)


plot_crosscorrelation(juxta_times, timedata)




# How to plot the histograms in space for 128channels?


# Return all_electrodes

def create_128channels_imec_prb(filename=None, bad_channels=None):

    r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1,61,	57,
                   36,	34,	32,	30,	28,	26,	24,	22,	20])
    r2 = np.array([106, 104, 115, 117, 119, 121, 123, 125, 127, 71, 67, 74, 76, 78, 114, 112,
                   49, 51, 53, 55, 2, 62, 58, 4, 6, 8, 10, 12, 14, 21, 19, 16])
    r3 = np.array([102, 100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59,
                   39, 37, 35, 33, 31, 29, 27, 25, 23])
    r4 = np.array([109, 107, 105, 116, 118, 120, 122, 124, 126, 73, 69, 64, 75, 77, 79, 113,
                   48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18,-1])

    all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
    all_electrodes = all_electrodes_concat.reshape((4, 32))
    all_electrodes = np.flipud(all_electrodes.T)

    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes)

    return all_electrodes


def open_hdf5_kwx(filename):

    f = h5py.File(filename,'r')
    dataset = f['/channel_groups/0/features_masks']
    #dataset = f['/channel_groups/1/features_masks']
    dataset.shape
    teste= dataset[:,:,1]
    #gh=teste[:,slice(0,96,3)]
    gh=teste[:,slice(0,384,3)]
    return np.array(gh)



def open_hdf5_kw(filename):
    f = h5py.File(filename,'r')
    times = f['/channel_groups/0/spikes/time_samples']
    #times = f['/channel_groups/1/spikes/time_samples']
    return np.array(times)


def times_spikes_ch(maskdata,timedata):
    global time_ch

    time_ch=[]
    for i in np.arange(128):
        maskdata_ch=maskdata[:,i]
        t=[timedata[x] for x in range(np.shape(timedata)[0]) if maskdata_ch[x] > 0]
        time_ch.append(t)
        i=i+1



cellname= '2015_09_03_pair9.0'
maskdata= open_hdf5_kwx(r'D:\Protocols\PairedRecordings\Neuroseeker128\Data\2015-09-03\KLUSTA_2015_09_03_Pair_9_0\threshold_4_5std_v2.kwx')
timedata= open_hdf5_kw(r'D:\Protocols\PairedRecordings\Neuroseeker128\Data\2015-09-03\KLUSTA_2015_09_03_Pair_9_0\threshold_4_5std_v2.kwik')
times_spikes_ch(maskdata,timedata)#return time_ch

juxta_times= all_cells_spike_triggers['9'] # first, run the paired_128ch with the cellname pair

all_electrodes = create_128channels_imec_prb()
electrode_structure=all_electrodes


def plot_crosscorrelation(juxta,timematriz, ysup=5500, x=50):
    global cross
    cross = []
    juxta=np.asarray(juxta, dtype=np.float64)
    for i in np.arange(128):
        channel= (np.array(timematriz[i])).astype(np.float64)
        cross_array= crosscorrelate(channel, juxta, lag=1500)
        cross.append(cross_array)

    subplot_number_array = electrode_structure.reshape(1,128)
    for i in np.arange(1, subplot_number_array.shape[1]+1):
        plt.subplot(32,4,i)
        plt.hist((cross[np.int(subplot_number_array[:,i-1])][0])/30,bins=101, range=(-50,50), align='mid')
        plt.ylim(0, ysup)
        plt.xlim(-x,x)


plot_crosscorrelation(juxta_times, time_ch)

# MAX of detections for each channel
max=[]
index=[]
plt.figure()
for i in np.arange(128):
    n,b,p=plt.hist((cross[i][0])/30,bins=101, range=(-50,50), align='mid')
    max.append(n.max())
    index.append(n.argmax())

max = np.array(max)
index = np.array(index)

print(max.max())
print(max.argmax())





####cross w times from spikes

def plot_crosscorrelation(juxta,timedata, x=50):
    global cross
    cross = []
    juxta=np.asarray(juxta, dtype=np.float64)
    channel= (np.array(timedata)).astype(np.float64)
    cross = crosscorrelate(channel, juxta, lag=1500)
    plt.figure()
    n,b,p=plt.hist((cross[0])/30,bins=101, range=(-50,50), align='mid')
    max = n.max()
    index= n.argmax()
    print(max.max())
    print(max.argmax())
    #plt.ylim(0, ysup)
    plt.xlim(-x,x)


plot_crosscorrelation(juxta_times, timedata)



