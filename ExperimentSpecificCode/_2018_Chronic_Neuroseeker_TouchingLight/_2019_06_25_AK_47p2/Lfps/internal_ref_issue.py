
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
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
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
lfp_average_data_folder = join(results_folder, 'Lfp', 'Averages')

lfps_filename = join(data_folder, 'Amplifier_LFPs.bin')

trial_pokes_timepoints_filename = join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')
non_trial_pokes_timepoints_filename = join(events_definitions_folder,
                                           'events_first_pokes_after_5_delay_non_reward.npy')

# Load data
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

lfps = ns_funcs.load_binary_amplifier_data(lfps_filename, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

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


# -------------------------------------------------
# <editor-fold desc="SEE LFPS IN RESPECT TO THE REFERENCE CHANNELS">

closest_reference_positions_bottom = np.arange(2, 72, 6) + 0.5
event_choise = 'ntp'
events = possible_events[event_choise]
num_of_events = len(events)

avg_lfps_around_event, std_lfps_around_event, time_axis = \
    tla_funcs.time_lock_raw_data(lfps, events, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=None,
                                 high_pass_cutoff=None, rectify=None, low_pass_cutoff=100,
                                 avg_reref=True, keep_trials=False)

smooth_factor = 1000
smoothed_avg_lfps_around_event = binning.rolling_window_with_step(avg_lfps_around_event, np.mean, smooth_factor, smooth_factor)
smoothed_time_x_axis = np.arange(-window_timepoints / const.SAMPLING_FREQUENCY,
                   window_timepoints/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY * smooth_factor)

plt.figure(2)
plt.imshow(smoothed_avg_lfps_around_event, aspect='auto')
plt.hlines(y=closest_reference_positions_bottom, xmin=0, xmax=smoothed_avg_lfps_around_event.shape[1])
plt.hlines(y=np.arange(0, 72, 6)-0.5, xmin=0, xmax=smoothed_avg_lfps_around_event.shape[1], color='r')


# </editor-fold>
# -------------------------------------------------