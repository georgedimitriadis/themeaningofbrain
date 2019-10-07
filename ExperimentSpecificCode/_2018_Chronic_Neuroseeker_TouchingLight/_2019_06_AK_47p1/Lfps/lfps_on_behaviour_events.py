
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis import timelocked_analysis_functions as tla_funcs

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

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')

lfps_filename = join(data_folder, 'Amplifier_LFPs.bin')

trial_pokes_timepoints_filename = join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')
non_trial_pokes_timepoints_filename = join(events_definitions_folder,
                                           'events_first_pokes_after_5_delay_non_reward.npy')

# Load data
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)

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
event_choise = 'ntp'
events = possible_events[event_choise]
num_of_events = len(events)

avg_lfps_around_event, time_axis = \
    tla_funcs.time_lock_raw_data(lfps, events, times_to_cut=[-window_time, window_time],
                                 sampling_freq=const.SAMPLING_FREQUENCY,
                                 baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=None,
                                 high_pass_cutoff=None, rectify=None, low_pass_cutoff=100,
                                 avg_reref=True, keep_trials=False)

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
good_channels = np.arange(12, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, 1)
dataset = smoothed_avg_lfps_around_event[good_channels, :] #  z_score,  smoothed_avg_lfps_around_event
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
plt.imshow(np.flipud(dataset[np.arange(2, 60, 2)]), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[0], lfp_probe_positions[-2]],
           interpolation=interpolation)
plt.hlines(y=pos, xmin=-8, xmax=8)
plt.title(title)

f = plt.figure(3)
plt.imshow(np.flipud(dataset[np.arange(2, 60, 2)]), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[1], lfp_probe_positions[-1]],
           interpolation=interpolation)
plt.hlines(y=pos, xmin=-8, xmax=8)
plt.title(title)


# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="LFPS AROUND SPIKES">

event_type = 'tp'
mod_type = 'i'

events = possible_events[event_type]
neurons = neuron_indices[event_type][mod_type]

# Use the all spikes of a neuron
lfps_around_all_spikes = {}
for n in neurons:
    spike_times = template_info.iloc[n]['spikes in template']
    num_of_events = len(spike_times)
    temp = np.empty((const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, num_of_events, 2 * window_timepoints))
    for i in np.arange(num_of_events):
        temp[:, i, :] = lfps[:, spike_times[i] - window_timepoints:
                                spike_times[i] + window_timepoints]
    lfps_around_all_spikes[n] = np.mean(temp, axis=1)
    print(n)


_ = plt.plot(space(lfps_around_all_spikes[neurons[8]]).T)


# Use the spike closest to the event
lfps_around_event_spikes = {}
for n in neurons:
    spike_times = template_info.iloc[n]['spikes in template']
    event_spike_times = []
    for ev in events:
        event_spike_times.append(spike_times[np.argmin(np.abs(spike_times-ev))])
    num_of_events = len(event_spike_times)
    temp = np.empty((const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, num_of_events, 2 * window_timepoints))
    for i in np.arange(num_of_events):
        temp[:, i, :] = lfps[:, event_spike_times[i] - window_timepoints:
                                          event_spike_times[i] + window_timepoints]
    lfps_around_event_spikes[n] = np.mean(temp, axis=1)
    print(n)

def show(index, ax):
    ax.clear()
    ax.plot(space(lfps_around_event_spikes[neurons[index]]).T)
    return None

index = 0
fig = plt.figure(0)
ax = fig.add_subplot(111)
out = None
args = [ax]
sl.connect_repl_var(globals(), 'index', 'out', 'show', 'args', slider_limits=[0, len(neurons)-1])

# </editor-fold>
# -------------------------------------------------
