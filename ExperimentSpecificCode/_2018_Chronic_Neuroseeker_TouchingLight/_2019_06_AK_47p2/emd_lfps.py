


import numpy as np
import pandas as pd
from os.path import join
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import matplotlib.pyplot as plt
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const

from BrainDataAnalysis.LFP import emd

import sequence_viewer as seq_v
import transform as tr
import one_shot_viewer as one_v
import slider as sl

from BrainDataAnalysis.LFP import emd


# ----------------------------------------------------------------------------------------------------------------------
# LOAD FOLDER AND DATA
date = 8
data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date], 'Data')
binary_data_filename = join(data_folder, 'Amplifier_LFPs.bin')

sampling_freq = const.SAMPLING_FREQUENCY

raw_lfp = ns_funcs.load_binary_amplifier_data(binary_data_filename, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)


# ----------------------------------------------------------------------------------------------------------------------
# HAVE A LOOK AT THE LFPS
# ----------------------------------------------------------------------------------------------------------------------
raw_lfp = raw_lfp[:, :89240000]
time_points_buffer = 5000
lfp_data_panes = np.swapaxes(
    np.reshape(raw_lfp, (raw_lfp.shape[0], int(raw_lfp.shape[1] / time_points_buffer), time_points_buffer)), 0, 1)

lfp_channels_on_probe = np.arange(9, 1440, 20)
channels_heights = ns_funcs.get_channels_heights_for_spread_calulation(lfp_channels_on_probe)
bad_lfp_channels = []
lfp_channels_used = np.delete(np.arange(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE), bad_lfp_channels)

pane = 100

def spread_lfp_pane(p):
    pane = lfp_data_panes[p, :, :]
    spread = ns_funcs.spread_data(pane, channels_heights, lfp_channels_used)
    spread = np.flipud(spread)
    return spread


pane_data = None
tr.connect_repl_var(globals(), 'pane', 'spread_lfp_pane', 'pane_data')

one_v.graph(globals(), 'pane_data')

def do_nothing(p):
    return p

nothing = None
slider_limits = [0, lfp_data_panes.shape[0]-1]
sl.connect_repl_var(globals(), 'pane', 'do_nothing', 'nothing', slider_limits=slider_limits)


# ----------------------------------------------------------------------------------------------------------------------
# SUBSAMPLE THE LFPS WITH DIFFERENT RATIOS AND SAVE THE FILES
# ----------------------------------------------------------------------------------------------------------------------

ds_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_LFPs_Downsampled_x4.bin')

downsampled_lfp = ns_funcs.downsample(filename=ds_filename, data=raw_lfp, factor=4)


ds_numpy_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                         'Analysis', 'Lfp', 'Downsampling', 'Amplifier_LFPs_Downsampled_x4.npy')
np.save(ds_numpy_filename, downsampled_lfp)
downsampled_lfp = np.load(ds_numpy_filename)

# ----------------------------------------------------------------------------------------------------------------------
# HAVE A LOOK AT THE DOWSAMPLED DATA
# ----------------------------------------------------------------------------------------------------------------------

downsampled_lfp = np.load(ds_numpy_filename)

def space_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (500*i) for i in np.arange(dat.shape[0])]))
    return result


timepoint = 100000
buffer = 10000

seq_v.graph_range(globals(), 'timepoint', 'buffer', 'downsampled_lfp', transform_name='space_data')


# ----------------------------------------------------------------------------------------------------------------------
# RUN THE emd.py TO GENERATE THE IMFS
# ----------------------------------------------------------------------------------------------------------------------

'''
Parameters used
result_dtype = np.int16
num_imfs = 13
ensemble_size = 25
noise_strength = 0.01
S_number = 20
num_siftings = 100

cmd used
cd E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\BrainDataAnalysis\LFP
python emd.py "F:\Neuroseeker chronic\AK_47.2\2019_06_25-12_50\Analysis\Lfp\Downsampling\Amplifier_LFPs_Downsampled_x4.npy" "F:\Neuroseeker chronic\AK_47.2\2019_06_25-12_50\Analysis\Lfp\EMD\imfs.bin" int16 13 25 0.01 20 100
'''