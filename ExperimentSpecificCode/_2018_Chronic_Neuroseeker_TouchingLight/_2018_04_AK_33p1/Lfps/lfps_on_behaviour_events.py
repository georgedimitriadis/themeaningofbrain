
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
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
kilosort_folder = join(analysis_folder, 'Denoised', 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
lfp_average_data_folder = join(results_folder, 'Lfp', 'Averages')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')
patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')
regressions_folder = join(results_folder, 'Regressions')

ballistic_mov_folder = join(results_folder, 'EventsCorrelations', 'StartBallisticMovToPoke')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')

emd_filename = join(analysis_folder, 'Lfp', 'EMD', 'imfs.bin')
lfps_filename = join(data_folder, 'Amplifier_LFPs.bin')

trial_pokes_timepoints_filename = join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')
non_trial_pokes_timepoints_filename = join(events_definitions_folder,
                                           'events_first_pokes_after_5_delay_non_reward.npy')
start_balistic_frames_filename = join(ballistic_mov_folder, 'start_of_ballistic_traj_frames_hand_picked.npy')

# Load data
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cortex_sorting.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)

distances_rat_to_poke_all_frames = np.load(join(patterned_vs_non_patterned_folder,
                                                'distances_rat_to_poke_all_frames.npy'))

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
# <editor-fold desc="LFPS AROUND EVENTS">
event_choise = 'tp'
events = possible_events[event_choise]
num_of_events = len(events)

avg_lfps_around_event, time_axis = \
    tla_funcs.time_lock_raw_data(lfps, events, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=None,
                                 high_pass_cutoff=None, rectify=None, low_pass_cutoff=100,
                                 avg_reref=True, keep_trials=False)

#avg_lfps_around_event = -avg_lfps_around_event
'''
imfs_around_tp = np.empty((const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, const.NUMBER_OF_IMFS,
                           num_of_events, int(2 * window_timepoints / const.LFP_DOWNSAMPLE_FACTOR)))
for i in np.arange(num_of_events):
    start_imfs = int((events[i] - window_timepoints) / const.LFP_DOWNSAMPLE_FACTOR)
    end_imfs = int((events[i] + window_timepoints) / const.LFP_DOWNSAMPLE_FACTOR)
    imfs_around_tp[:, :, i, :] = imfs[:, :, start_imfs:end_imfs]
avg_imfs_around_tp = np.mean(imfs_around_tp, axis=2)
avg_imfs_around_tp = np.swapaxes(avg_imfs_around_tp, 0, 1)
'''


def space(data):
    return cdf.space_data_factor(data, 2)


imf = 0
sv.graph_pane(globals(), 'imf', 'avg_imfs_around_tp', transform_name='space')

_ = plt.plot(space(avg_lfps_around_event).T)



random_times = np.random.choice(np.arange(2*window_timepoints,
                                          lfps.shape[1]-2*window_timepoints,
                                          1),
                                num_of_events)
random_triggered_lfps = []
for spike in random_times:
    random_triggered_lfps.append(lfps[:, spike - window_timepoints:spike + window_timepoints])
random_triggered_lfps = np.array(random_triggered_lfps)
random_triggered_lfps_mean = random_triggered_lfps.mean(axis=0)
random_triggered_lfps_std = random_triggered_lfps.std(axis=0)

# _ = plt.plot(space(random_triggered_lfps_std).T)

z_score = []
for channel in np.arange(avg_lfps_around_event.shape[0]):
    z_score.append((avg_lfps_around_event[channel, :] - random_triggered_lfps_mean[channel, :]) /
                   (random_triggered_lfps_std[channel, :] / np.sqrt(len(events))))
z_score = np.array(z_score)

time_x_axis = np.arange(-window_timepoints/const.SAMPLING_FREQUENCY,
                   window_timepoints/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY)


_ = plt.plot(time_x_axis, z_score.T)
_ = plt.plot(time_x_axis, cdf.space_data(z_score, 3).T)

_ = plt.plot(time_x_axis[:-300:100], cdf.space_data(binning.rolling_window_with_step(z_score, np.mean, 300, 100), 3).T)
_ = plt.plot(time_x_axis[:-300:100], binning.rolling_window_with_step(z_score, np.mean, 300, 100).T)


index = 0
sv.graph_pane(globals(), 'index', 'z_factor', 'time_x_axis')


t = np.argwhere(np.abs(z_score) > 5)
channels_with_large_changes = np.unique(t[:, 0])


smooth_factor = 1000
#smoothed_z_score = binning.rolling_window_with_step(z_score, np.mean, smooth_factor, smooth_factor)
smoothed_avg_lfps_around_event = binning.rolling_window_with_step(avg_lfps_around_event, np.mean, smooth_factor, smooth_factor)
smoothed_time_x_axis = np.arange(-window_timepoints / const.SAMPLING_FREQUENCY,
                   window_timepoints/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY * smooth_factor)


# _ = plt.plot(smoothed_time_x_axis, smoothed_z_score[channels_with_large_changes, :].T)
# _ = plt.plot(smoothed_time_x_axis, cdf.space_data_factor(smoothed_z_score[channels_with_large_changes, :], 1.2).T)


pos = list(const.BRAIN_REGIONS.values())
dataset = smoothed_avg_lfps_around_event #  z_score,  smoothed_avg_lfps_around_event
interpolation = 'bilinear'
title = 'Rat = {}, Event = {}'.format(const.rat_folder, str(event_choise))
f = plt.figure(1)
plt.imshow(np.flipud(dataset), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[0], lfp_probe_positions[-1]],
           interpolation=interpolation)
plt.hlines(y=pos, xmin=-8, xmax=8)
plt.title(title)

f = plt.figure(2)
plt.imshow(np.flipud(dataset[np.arange(0, 72, 2)]), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[0], lfp_probe_positions[-2]],
           interpolation=interpolation)
plt.hlines(y=pos, xmin=-8, xmax=8)
plt.title(title)

f = plt.figure(3)
plt.imshow(np.flipud(dataset[np.arange(1, 72, 2)]), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[1], lfp_probe_positions[-1]],
           interpolation=interpolation)
plt.hlines(y=pos, xmin=-8, xmax=8)
plt.title(title)


# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="COMPARE LFPS AROUND THE SUCCESSFUL AND THE NON_SUCCESSFUL EVENTS">

# <editor-fold desc="Create the data sets (RUN ONCE AND SAVE)">
event_choise = 'tp'
tps = possible_events[event_choise]
num_of_tp = len(tps)

lfps_around_tp, avg_lfps_around_tp, std_lfps_around_tp, time_axis = \
    tla_funcs.time_lock_raw_data(lfps, tps, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=None,
                                 high_pass_cutoff=None, rectify=None, low_pass_cutoff=100,
                                 avg_reref=True, keep_trials=True)


event_choise = 'ntp'
ntps = possible_events[event_choise]
num_of_ntp = len(ntps)

lfps_around_ntp, avg_lfps_around_ntp, std_lfps_around_tp, time_axis = \
    tla_funcs.time_lock_raw_data(lfps, ntps, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=None,
                                 high_pass_cutoff=None, rectify=None, low_pass_cutoff=100,
                                 avg_reref=True, keep_trials=True)

smooth_factor = 1000
lfps_around_tp_smooth = np.empty((lfps_around_tp.shape[0], int(lfps_around_tp.shape[1] / smooth_factor),
                                  lfps_around_tp.shape[2]))
lfps_around_ntp_smooth = np.empty((lfps_around_ntp.shape[0], int(lfps_around_ntp.shape[1] / smooth_factor),
                                  lfps_around_ntp.shape[2]))
for i in np.arange(lfps_around_tp.shape[2]):
    lfps_around_tp_smooth[:, :, i] = binning.rolling_window_with_step(lfps_around_tp[:, :, i], np.mean, smooth_factor, smooth_factor)
    lfps_around_ntp_smooth[:, :, i] = binning.rolling_window_with_step(lfps_around_ntp[:, :, i], np.mean, smooth_factor, smooth_factor)

np.save(join(lfp_average_data_folder, 'lfps_around_tp_smooth.npy'), lfps_around_tp_smooth)
np.save(join(lfp_average_data_folder, 'lfps_around_ntp_smooth.npy'), lfps_around_ntp_smooth)
# </editor-fold>

lfps_around_tp_smooth = np.load(join(lfp_average_data_folder, 'lfps_around_tp_smooth.npy'))
lfps_around_ntp_smooth = np.load(join(lfp_average_data_folder, 'lfps_around_ntp_smooth.npy'))

lfps_around_tp_smooth_left = lfps_around_tp_smooth[np.arange(0, 72, 2)]
lfps_around_tp_smooth_right = lfps_around_tp_smooth[np.arange(1, 72, 2)]
lfps_around_ntp_smooth_left = lfps_around_ntp_smooth[np.arange(0, 72, 2)]
lfps_around_ntp_smooth_right = lfps_around_ntp_smooth[np.arange(1, 72, 2)]


p_values_right, cluster_labels_under_alpha_right = \
    cl_per.monte_carlo_significance_probability(lfps_around_tp_smooth_right, lfps_around_ntp_smooth_right,
                                                num_permutations=1000, min_area=10, cluster_alpha=0.05,
                                                monte_carlo_alpha=0.01, sample_statistic='independent',
                                                cluster_statistic='maxsum')

p_values_left, cluster_labels_under_alpha_left = \
    cl_per.monte_carlo_significance_probability(lfps_around_tp_smooth_left, lfps_around_ntp_smooth_left,
                                                num_permutations=1000, min_area=10, cluster_alpha=0.05,
                                                monte_carlo_alpha=0.01, sample_statistic='independent',
                                                cluster_statistic='maxsum')


pos = list(const.BRAIN_REGIONS.values())
data = lfps_around_tp_smooth_left.mean(-1)
cluster_labels = cluster_labels_under_alpha_left
plf.show_significant_clusters_on_data(data, cluster_labels, pos, lfp_probe_positions, window_time=8)

data = lfps_around_tp_smooth_right.mean(-1)
cluster_labels = cluster_labels_under_alpha_right
plf.show_significant_clusters_on_data(data, cluster_labels, pos, lfp_probe_positions, window_time=8)

# </editor-fold>
# -------------------------------------------------

