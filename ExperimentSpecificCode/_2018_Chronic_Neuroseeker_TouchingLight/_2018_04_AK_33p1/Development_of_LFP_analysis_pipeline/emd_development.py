


import numpy as np
import pandas as pd
from os.path import join
from scipy.signal import welch
from mne.time_frequency import multitaper as mt
import nitime as ni
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import matplotlib.pyplot as plt
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from BrainDataAnalysis.LFP import emd

import sequence_viewer as seq_v
import transform as tr
import one_shot_viewer as one_v

import PyEMD
from BrainDataAnalysis.LFP import emd

import pyeemd
import time

# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
date = 8
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_LFPs.bin')

sampling_freq = const.SAMPLING_FREQUENCY


# Load data
raw_lfp = ns_funcs.load_binary_amplifier_data(binary_data_filename, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)


# SUBSAMPLE THE LFPS WITH DIFFERENT RATIOS AND SAVE THE FILES
'''
ds_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_LFPs_Downsampled_x4.bin')

downsampled_lfp = ns_funcs.downsample(filename=ds_filename, data=raw_lfp, factor=4)


ds_numpy_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                         'Analysis', '\Lfp', 'Downsampling', 'Amplifier_LFPs_Downsampled_x4.npy')
np.save(ds_numpy_filename, downsampled_lfp)
downsampled_lfp = np.load(ds_numpy_filename)
'''

factor = 4  # possible 3, 4, 5, 10
ds_numpy_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                         'Analysis', 'Lfp', 'Downsampling', 'Amplifier_LFPs_Downsampled_x{}.npy'.format(factor))
downsampled_lfp = np.load(ds_numpy_filename)



# HAVE A LOOK AT THE RAW DATA
'''
def space_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (500*i) for i in np.arange(dat.shape[0])]))
    return result


timepoint = 100000
buffer = 10000

seq_v.graph_range(globals(), 'timepoint', 'buffer', 'raw_lfp', transform_name='space_data')
'''


# TESTING DIFFERENT SPECTRAL DENSITY METHODS
'''
# Weltch is not half as good as multi taper. Done here for comparison only
def weltch_psd(data):
    fs, psd = welch(data,
                    fs=20000,  # sample rate
                    window='hanning',   # apply a Hanning window before taking the DFT
                    nperseg=10000,        # compute periodograms of 256-long segments of x
                    noverlap=5000,
                    detrend='constant')
    return fs, psd
'''

# CHECKING OUT THE ERROR IN THE MULTITAPER METHOD
'''
psd, fs = mt.psd_array_multitaper(raw_lfp[:, timepoint:timepoint+10000], sfreq=sampling_freq, fmin=2, fmax=250,
                                          bandwidth=5)

# Used the multi taper from nitime to get an estimate of the variance of the psd (jackknife). It is too small to bother
fs_ni, psd_ni, jackknife = ni.algorithms.spectral.multi_taper_psd(raw_lfp[:, timepoint:timepoint+10000],
                                                                   Fs=sampling_freq,
                                                                   BW=6, jackknife=True)
'''

# SCAN THROUGH THE DATA AND HAVE A LOOK AT THE RESULTING PSDS
'''

timepoint = 100000
buffer = 10000
psd = []
global fs
fs = []


def psd_and_space(point):
    global fs
    data = raw_lfp[:, point:point+buffer]
    psd, fs = mt.psd_array_multitaper(data, sfreq=sampling_freq, fmin=2, fmax=150, bandwidth=6, verbose=0)
    psd = np.array(([psd[i, :] + (10000000*i) for i in np.arange(psd.shape[0])]))

    return psd


def psd_only(point):
    global fs
    data = raw_lfp[:, point:point+buffer]
    psd, fs = mt.psd_array_multitaper(data, sfreq=sampling_freq, fmin=2, fmax=150, bandwidth=6, verbose=0)

    return psd/100000


tr.connect_repl_var(globals(), 'timepoint', 'psd_only', 'psd')


one_v.graph(globals(), 'psd', 'fs')

colormap = 'seismic'
image_levels = [0, 100]
one_v.image(globals(), 'psd', image_levels=image_levels, colormap=colormap)
'''


# EMD TESTING
begin_point = 1000000
end_point = 1020000
example_data = raw_lfp[0, begin_point:end_point]
example_data_ds = downsampled_lfp[0, int(begin_point/factor):int(end_point/factor)]


# TESTING OF PyEMD PYTHON MODULE: NOT NICE CODE, WAY TOO SLOW
'''
config = {'FIXE_H': 200, 'MAX_ITERATION': 100}
pyemd = PyEMD.EMD(extrema_detection="simple", spline_kind='akima', **config)
imfs = pyemd.emd(example_data_ds, max_imf=20)


pyeemd = PyEMD.EEMD(trials=100, noise_width=0.05, ext_EMD=None)
imfs = pyeemd.emd(example_data_ds, max_imf=20)


mine_emd = emd.emd(example_data_ds, nIMF=10, stoplim=0.1)
emd.findnextcomponent(example_data_ds, example_data_ds, t, 0.1)

fig = plt.figure(2)
N = int(len(imfs))
axs=[]
for n in np.arange(N):
    axs.append(fig.add_subplot(N, 1, n+1))
    if n==0:
        axs[n].plot(example_data_ds)
    else:
        axs[n].plot(imfs_c[n])
'''


# TESTING THE pyeemd MODULE: NICE AND FAST (C CODE)
'''
t0 = time.process_time()
imfs = pyeemd.emd(example_data_ds, num_imfs=15, S_number=5, num_siftings=5000)
print(time.process_time() - t0)

t0 = time.process_time()
imfs_c_e5_n2 = pyeemd.ceemdan(example_data, num_imfs=15, ensemble_size=5, noise_strength=0.2, S_number=8, num_siftings=5000)
print(time.process_time() - t0)


data = real_imfs[:, :100000]
fig = plt.figure(2)
N = int(len(data))
axs=[]
for n in np.arange(N):
    axs.append(fig.add_subplot(N, 1, n+1))
    axs[n].plot(data[n])
    #axs[n].plot(example_data, color='0.6')


psd_imf, fs = mt.psd_array_multitaper(imfs_first, sfreq=sampling_freq/factor, fmin=1, fmax=3000, bandwidth=6, verbose=0)
psd_imf_c_e5_n2, fs = mt.psd_array_multitaper(imfs_c_e5_n2, sfreq=sampling_freq, fmin=1, fmax=5000, bandwidth=6, verbose=0)

data = psd_imf
fig = plt.figure(3)
N = int(len(data))
axs=[]
for n in np.arange(N):
    axs.append(fig.add_subplot(N, 1, n+1))
    axs[n].plot(fs, data[n])


t0 = time.process_time()
imfs = pyeemd.emd(downsampled_lfp[0, :], num_imfs=15, S_number=5, num_siftings=5000)
print(time.process_time() - t0)


# TESTING HOW TO DO EMD IN CHUNKS
begin_point = 1000000
end_point = 1020000
offset = 10000
ds_first = downsampled_lfp[0, int(begin_point/factor):int(end_point/factor)]
ds_second = downsampled_lfp[0, int((begin_point + offset)/factor):int((end_point + offset)/factor)]

num_imfs = 13
ensemble_size = 25
noise_strength = 0.01
S_number = 20
num_siftings = 100
'''

# EMD not working. Too variable
'''
imfs_first = pyeemd.emd(ds_first, num_imfs=num_imfs,
                            S_number=S_number, num_siftings=num_siftings)
imfs_second = pyeemd.emd(ds_second, num_imfs=num_imfs,
                             S_number=S_number, num_siftings=num_siftings)
'''

# CEEMDAN works nice but is slow
'''
t0 = time.process_time()
imfs_first = pyeemd.ceemdan(ds_first, num_imfs=num_imfs, ensemble_size=ensemble_size, noise_strength=noise_strength,
                            S_number=S_number, num_siftings=num_siftings)
print(time.process_time() - t0)

t0 = time.process_time()
imfs_second = pyeemd.ceemdan(ds_second, num_imfs=num_imfs, ensemble_size=ensemble_size, noise_strength=noise_strength,
                             S_number=S_number, num_siftings=num_siftings)
print(time.process_time() - t0)
# -------------------------


first_overlap_point = int(end_point/factor) - int((begin_point + offset)/factor)
second_overlap_point = int(end_point/factor) -  int(begin_point/factor)

overlap_points = int(end_point/factor) - int((begin_point + offset)/factor)
t = np.arange(overlap_points)

imf = 10
plt.plot(t, imfs_first[imf, -overlap_points:],
         t, imfs_second[imf, :overlap_points])


t = np.arange(0, test_imfs.shape[1]*factor/sampling_freq, factor/sampling_freq)
imf = 8
plt.plot(t, test_imfs[imf], color='b')
plt.plot(t, real_imfs[imf], color='r')
'''


# Load the generated EMD data and have a look
num_of_imfs = const.NUMBER_OF_IMFS
num_of_channels = const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE

imfs_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                         'Analysis', 'Lfp', 'EMD', 'imfs.bin')

imfs = emd.load_memmaped_imfs(imfs_filename, dtype=np.int16, num_of_imfs=num_of_imfs, num_of_channels=num_of_channels)


import one_shot_viewer as osv
import slider as sl
import common_data_transforms as cdts

imf = 0
time = 1000000
buffer = 5000
factor = 2
args = [factor, time, buffer, imfs]


def get_specific_imf_spaced(imf, factor, data):
    imf_data = data[:, imf, :]
    return cdts.space_data_factor(imf_data, factor)


def get_time_window(time, buffer, data):
    return data[:, :, time:time+buffer]


def get_windowed_imf(imf, factor, time, buffer, data):
    return get_specific_imf_spaced(imf, factor, get_time_window(time, buffer, data))


def update_args(t):
    return [factor, t, buffer, imfs]


sl.connect_repl_var(globals(), 'time', 'update_args', 'args', slider_limits=[1000000, 1200000])


window = None


sl.connect_repl_var(globals(), 'imf', 'get_windowed_imf', 'window', 'args', slider_limits=[0, 12])


osv.graph(globals(), 'window')


# Checking imfs similarity
t = imfs[30, :, :]
for i in np.arange(t.shape[0]-1):
    vA = t[i, :]
    vB = t[i+1, :]
    len_vA = np.linalg.norm(vA)
    len_vB = np.linalg.norm(vB)
    cos = np.dot(vA, vB) / (len_vA * len_vB)
    print(cos)


