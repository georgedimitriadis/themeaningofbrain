

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BrainDataAnalysis.LFP import emd
import common_data_transforms as cdts
from mne.time_frequency import multitaper as mt

import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs


import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
#  -------------------------------------------------
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')
lfp_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder],
                         'Analysis', 'Lfp')
imfs_filename = join(lfp_folder, 'EMD', 'imfs.bin')

imf_kilosort_folder = join(lfp_folder, 'Kilosort')

spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

#  -------------------------------------------------
# Load the generated EMD data and have a look
#  -------------------------------------------------
num_of_imfs = const.NUMBER_OF_IMFS
num_of_channels = const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE


imfs = emd.load_memmaped_imfs(imfs_filename, dtype=np.int16, num_of_imfs=num_of_imfs, num_of_channels=num_of_channels)

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

#  -------------------------------------------------
# TEST WHAT HAPPENS IF I KILOSORT ONE IMF OF ALL CHANNELS
#  -------------------------------------------------

# Prepare the data
imf_for_ks = 3
imf_for_ks_filename = join(imf_kilosort_folder, 'imf_{}'.format(imf_for_ks), 'single_imf_{}.bin'.format(imf_for_ks))
'''
imf_data_for_ks = np.memmap(imf_for_ks_filename, dtype=imfs.dtype, mode='w+', shape=(imfs.shape[0], imfs.shape[2]))
imf_data_for_ks[:, :] = np.array(imfs[:, imf_for_ks, :].copy())
del imf_data_for_ks
'''
imf_data_for_ks = np.memmap(imf_for_ks_filename, mode='r', dtype=imfs.dtype, shape=(imfs.shape[0], imfs.shape[2]))
channel = 0
start = 1000000
end = 1100000
f = plt.figure(2)
args = [f]
out = None
def show_one_channel(channel, figure):
    figure.clear()
    ax = figure.add_subplot(111)
    ax.plot(imf_data_for_ks[channel, start:end])
    return None

sl.connect_repl_var(globals(), 'channel', 'show_one_channel', 'out', 'args', [0, 71])
#  -------------------------------------------------

avg_spike_template = np.load(join(imf_kilosort_folder, 'imf_{}'.format(imf_for_ks), 'avg_spike_template.npy'))


imf_3_panes = np.swapaxes(np.reshape(imf_data_for_ks, (72, 1000, 18161)), 0, 1)
t = 0
image_levels = [0, 255]
sv.image_sequence(globals(), 't', 'imf_3_panes', image_levels=image_levels)