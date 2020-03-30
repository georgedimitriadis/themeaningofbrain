
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis.Statistics import binning, cluster_based_permutation_tests as cl_per
from BrainDataAnalysis import timelocked_analysis_functions as tla_funcs
from BrainDataAnalysis.Graphics import ploting_functions as plf


import pandas as pd
import matplotlib.pyplot as plt

import sequence_viewer as sv
import slider as sl
import common_data_transforms as cdf


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
# Folder definitions
date_folder = 6

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
lfp_average_data_folder = join(results_folder, 'Lfp', 'Averages')

ap_data_filename = join(data_folder, 'Amplifier_APs.bin')

trial_pokes_timepoints_filename = join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')
non_trial_pokes_timepoints_filename = join(events_definitions_folder,
                                           'events_first_pokes_after_5_delay_non_reward.npy')

# Load data

ap_data = ns_funcs.load_binary_amplifier_data(ap_data_filename, const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

trial_pokes_timepoints = np.load(trial_pokes_timepoints_filename)
non_trial_pokes_timepoints = np.load(non_trial_pokes_timepoints_filename)

window_time = 8
window_timepoints = int(window_time * const.SAMPLING_FREQUENCY)
window_downsampled = int(window_timepoints / const.LFP_DOWNSAMPLE_FACTOR)

possible_events = {'tp': trial_pokes_timepoints,
                   'ntp': non_trial_pokes_timepoints}

lfp_probe_positions = np.empty(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)
lfp_probe_positions[np.arange(0, 72, 2)] = (((np.arange(9, 1440, 40) + 1) / 4).astype(np.int) + 1) * 22.5
lfp_probe_positions[np.arange(1, 72, 2)] = (((np.arange(29, 1440, 40) + 1) / 4).astype(np.int)) * 22.5
# </editor-fold>
# -------------------------------------------------

#  Make and save the MUAe
event_choise = 'ntp'
events = possible_events[event_choise]
num_of_events = len(events)


avg_muae_around_ntp, std_muae_around_ntp, time_axis = \
    tla_funcs.time_lock_raw_data(ap_data, events, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=300,
                                 high_pass_cutoff=3000, rectify=True, low_pass_cutoff=400,
                                 avg_reref=True, keep_trials=False)

np.save(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_tp.npy'), avg_muae_around_tp)
np.save(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_ntp.npy'), avg_muae_around_ntp)


#  Normalise

avg_muae_around_tp = np.load(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_tp.npy'))
avg_muae_around_ntp = np.load(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_ntp.npy'))


regions_pos = np.array(list(const.BRAIN_REGIONS.values()))
pos_to_elect_factor = const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE / 8100
region_lines = binning.scale(regions_pos, np.min(regions_pos) * pos_to_elect_factor, np.max(regions_pos) * pos_to_elect_factor)

_ = plt.figure(1)
plt.imshow(np.flipud(avg_muae_around_tp), aspect='auto')
plt.vlines(x=avg_muae_around_tp.shape[1] / 2, ymin=0, ymax=avg_muae_around_tp.shape[0] - 1)
plt.hlines(y=avg_muae_around_tp.shape[0] - region_lines, xmin=0, xmax=avg_muae_around_tp.shape[1]-1, linewidth=3, color='w')

tp_n = np.empty((avg_muae_around_ntp.shape))
for i in np.arange(len(avg_muae_around_ntp)):
    tp_n[i, :] = binning.scale(avg_muae_around_ntp[i], 0, 1)

_= plt.figure(2)
plt.imshow(np.flipud(tp_n), aspect='auto')
plt.vlines(x=tp_n.shape[1] / 2, ymin=0, ymax=tp_n.shape[0] - 1)
plt.hlines(y=tp_n.shape[0] - region_lines, xmin=0, xmax=tp_n.shape[1]-1, linewidth=1, color='w')

tp_n_smooth = binning.rolling_window_with_step(tp_n, np.mean, 40, 40)

_= plt.figure(3)
plt.imshow(np.flipud(tp_n_smooth), aspect='auto')
plt.vlines(x=tp_n_smooth.shape[1] / 2, ymin=0, ymax=tp_n_smooth.shape[0] - 1)
plt.hlines(y=tp_n_smooth.shape[0] - region_lines, xmin=0, xmax=tp_n_smooth.shape[1]-1, linewidth=1, color='w')

_= plt.figure(3)
plt.imshow(np.flipud(tp_n_smooth), aspect='auto', extent=[-8, 8, len(tp_n_smooth), 0])
plt.hlines(y=tp_n_smooth.shape[0] - region_lines, xmin=-8, xmax=8, linewidth=3, color='w')
plt.vlines(x=0, ymin=0, ymax=tp_n_smooth.shape[0] - 1)
