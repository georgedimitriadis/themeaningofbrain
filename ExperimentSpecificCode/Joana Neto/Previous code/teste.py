
import IO.ephys as ephys
import os
import numpy as np
import pandas as pd
import itertools
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
import BrainDataAnalysis.filters as filters
import time
from BrainDataAnalysis import Constants as ct



#Neuroseeker 256 channels

#256ch probe in saline 2017-02-08
raw_data_file_ivm = r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline\amplifier2017-02-07T18_12_22_int16.bin'
num_ivm_channels = 256
amp_dtype = np.int16


#256ch probe recording 2017-02-08

raw_data_file_ivm = r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512\amplifier2017-02-08T15_34_04\amplifier2017-02-08T15_34_04_int16.bin'
num_ivm_channels = 256
amp_dtype = np.int16


sampling_freq = 20000
high_pass_freq = 250
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

samples = 3000000 #2.5minutes


raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix [:, 0:samples]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)

def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered = highpass(temp_unfiltered, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size



def polytrode_256channels(bad_channels=[]):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''

    electrode_coordinate_grid = list(itertools.product(np.arange(0, 18),
                                                       np.arange(0, 15)))

    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\chanmap256.csv', delimiter=",")

    electrode_amplifier_index_on_grid = electrode_amplifier_index_on_grid.astype(np.int16)


    reshaped = np.reshape(electrode_amplifier_index_on_grid,np.shape(electrode_amplifier_index_on_grid)[0]*np.shape(electrode_amplifier_index_on_grid)[1])

    electrode_amplifier_index_on_grid = reshaped
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))

    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)

    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions



# bad channels:
bad_channels = [7, 8, 19, 28, 30, 32, 34, 94, 131, 137, 148, 149, 151, 162, 171, 232, 31]
channels = np.delete(np.arange(num_ivm_channels), bad_channels)


#data_folder = '/home/jesse/Data/recording/2017_03_29-14_57/'
data_folder = '/home/jesse/Data/recording/2017_03_23_recording_probe_272'
fn = 'amplifier2017-03-23T22_02_33.bin'

raw_data_file = os.path.join(data_folder, fn)
raw_data = ephys.load_raw_data(raw_data_file, numchannels=num_channels, dtype=amp_dtype)



def get_average_corrmat_whole_data(data,show_heatmap=True):

    data_shape = data.shape
    chunk_size = 500000
    n_chunks = int(data_shape[1]/chunk_size)

    corrmat = np.empty((num_ivm_channels,num_ivm_channels,n_chunks))

    for i in np.arange(0,n_chunks):
        print('Working on chunk '+str(i+1)+' out of '+str(n_chunks))
        chunk = data[:,i*chunk_size:chunk_size+i*chunk_size]
        corrmat[:,:,i] = get_corrmat(chunk, show_heatmap=False)

    corrmat = np.mean(corrmat,axis=2)
    if show_heatmap:
        plot_corr_heatmap(corrmat)

    return corrmat


def get_corrmat(data, show_heatmap=False):
    """
    This function computes the correlation matrix between the 180 channels of the probe.

    :param data:
    :param show_heatmap:
    :return:
    """

    channel_positions = get_channel_positions()
    data = data[channel_positions.index]

    data = lowpass_filter_in_chunks(data, chunk_size=10000)

    K = data.shape[0]
    corrmat = np.empty((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            corrmat[i,j] = pearsonr_in_chunks(data[i],data[j])

    if show_heatmap:
        plot_corr_heatmap(corrmat)

    return corrmat

def plot_corr_heatmap(corrmat):
    """
    This function plots a heatmap for correlations between
    :param corrmat:
    :return:
    """
    channel_positions = get_channel_positions()
    f, ax = plt.subplots(figsize=(12, 9))
    hm = sns.heatmap(corrmat,
                     square=True,
                     cmap='RdBu_r'
                     )
    hm.set_xticklabels(labels=channel_positions.values, rotation='vertical')
    hm.set_yticklabels(labels=channel_positions.values, rotation='horizontal')
    for ind, label in enumerate(hm.get_xticklabels()):
        if ind % 2 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(hm.get_yticklabels()):
        if ind % 2 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for i in np.arange(1, corrmat.shape[0], 11):
        ax.axhline(i - 1, color="w")
        ax.axvline(i - 1, color="w")
    f.tight_layout()

def pearsonr_in_chunks(x,y):
    """
    This function computes the pairwise pearson r over the rows of x and y and returns
    the mean correlation coefficient over all rows.

    :param x:
    :param y:
    :return:
    """

    r = pairwise_pearsonr(x,y,axis=0)
    return np.mean(r) # this is the mean over chunks


def pairwise_pearsonr(x, y, axis=0):
    """
    This function computes row or column wise pearson correlation coefficients
    of x and y

    :param x:
    :param y:
    :param axis:
    :return:
    """
    xm = x - np.mean(x, axis=axis, keepdims=True)
    ym = y - np.mean(y, axis=axis, keepdims=True)
    r_num = np.sum(xm * ym, axis=axis)
    r_den = np.sqrt((xm*xm).sum(axis=axis) * (ym*ym).sum(axis=axis))
    return r_num / r_den


def take_filtered_chunk(start_sample = 0, end_sample = 30000):
    chunk = low_pass_filter(raw_data.dataMatrix[:,start_sample:end_sample],
                                    30000,
                                    200,
                                    filterType='but',
                                    filterOrder=3,
                                    filterDirection='twopass')
    return chunk

