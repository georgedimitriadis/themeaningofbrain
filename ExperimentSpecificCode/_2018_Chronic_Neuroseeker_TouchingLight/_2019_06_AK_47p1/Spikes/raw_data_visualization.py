
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt

from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
import common_data_transforms as cdt

from spikesorting_tsne_guis import clean_kilosort_templates as clean
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p1 import constants as const

import sys
from io import StringIO

import transform as tr
import sequence_viewer as sv


# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
date = 6
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis')

sampling_freq = const.SAMPLING_FREQUENCY

raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename, const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

raw_data = raw_data[240:, :]


timepoint_step = 1000

timepoint = 0

data_thal = raw_data[:4*120, :]
data_thal_bottom = raw_data[:2*120, :]
data_ca3 = raw_data[4*120:6*120, :]
data_ca1 = raw_data[6*120:8*120, :]
data_cort = raw_data[8*120:, :]

data_half = raw_data[:600, :]

def space(data):
    return cdt.space_data(data, 100)


sv.graph_range(globals(), 'timepoint', 'timepoint_step', 'data_thal', transform_name='space')

sv.graph_range(globals(), 'timepoint', 'timepoint_step', 'data_ca1', transform_name='space')

sv.graph_range(globals(), 'timepoint', 'timepoint_step', 'data_half', transform_name='space')


frame = 265
tp = 172000

start = 172200
end = 173200


start = 793200
end = start + 2000
plt.imshow(np.flipud(raw_data[:, start:end]), aspect='auto', cmap ='RdBu', vmin=-100, vmax=100)

plt.plot(np.flipud(cdt.space_data(raw_data[:, start:end], 100).T))