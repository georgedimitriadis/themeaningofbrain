
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
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
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

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
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

distances_rat_to_poke_all_frames = np.load(join(patterned_vs_non_patterned_folder,
                                                'distances_rat_to_poke_all_frames.npy'))

ti_increasing_neurons_on_trial_pokes = np.load(join(poke_folder, 'ti_increasing_neurons_on_trial_pokes.df'),
                                               allow_pickle=True)
ti_decreasing_neurons_on_trial_pokes = np.load(join(poke_folder, 'ti_decreasing_neurons_on_trial_pokes.df'),
                                               allow_pickle=True)
ti_increasing_neurons_on_non_trial_pokes = np.load(join(poke_folder,
                                                        'ti_increasing_neurons_on_non_trial_pokes.df'),
                                                   allow_pickle=True)
ti_decreasing_neurons_on_non_trial_pokes = np.load(join(poke_folder,
                                                        'ti_decreasing_neurons_on_non_trial_pokes.df'),
                                                   allow_pickle=True)
ti_decreasing_neurons_on_start_ballistic = np.load(join(ballistic_mov_folder, 'ti_decreasing_neurons_on_start_ballistic.npy'),
                                                   allow_pickle=True)
ti_increasing_neurons_on_start_ballistic = np.load(join(ballistic_mov_folder, 'ti_increasing_neurons_on_start_ballistic.npy'),
                                                   allow_pickle=True)

imfs = ns_funcs.load_imfs(emd_filename, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, const.NUMBER_OF_IMFS)
lfps = ns_funcs.load_binary_amplifier_data(lfps_filename, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

trial_pokes_timepoints = np.load(trial_pokes_timepoints_filename)
non_trial_pokes_timepoints = np.load(non_trial_pokes_timepoints_filename)
start_ballistic_frames = np.load(start_balistic_frames_filename)
start_ballistic_timepoints = ev_video['AmpTimePoints'].iloc[start_ballistic_frames].values

neuron_indices = {'tp': {'i': ti_increasing_neurons_on_trial_pokes.index.values,
                         'd': ti_decreasing_neurons_on_trial_pokes.index.values},
                  'ntp': {'i': ti_increasing_neurons_on_non_trial_pokes.index.values,
                          'd': ti_decreasing_neurons_on_non_trial_pokes.index.values},
                  'sb': {'i': ti_increasing_neurons_on_start_ballistic.index.values,
                         'd': ti_decreasing_neurons_on_start_ballistic.index.values}
                  }

window_time = 8
window_timepoints = int(window_time * const.SAMPLING_FREQUENCY)
window_downsampled = int(window_timepoints / const.LFP_DOWNSAMPLE_FACTOR)

possible_events = {'tp': trial_pokes_timepoints,
                   'ntp': non_trial_pokes_timepoints,
                   'sb': start_ballistic_timepoints}


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

imfs_around_events = np.empty((const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, const.NUMBER_OF_IMFS,
                               num_of_events, int(2 * window_timepoints / const.LFP_DOWNSAMPLE_FACTOR)))
for i in np.arange(num_of_events):
    start_imfs = int((events[i] - window_timepoints) / const.LFP_DOWNSAMPLE_FACTOR)
    end_imfs = int((events[i] + window_timepoints) / const.LFP_DOWNSAMPLE_FACTOR)
    imfs_around_events[:, :, i, :] = imfs[:, :, start_imfs:end_imfs]
avg_imfs_around_events = np.mean(imfs_around_events, axis=2)
avg_imfs_around_events = np.swapaxes(avg_imfs_around_events, 0, 1)



def space(data):
    return cdf.space_data_factor(data, 2)


imf = 0
sv.graph_pane(globals(), 'imf', 'avg_imfs_around_events', transform_name='space')
sv.image_sequence(globals(), 'imf', 'avg_imfs_around_events')

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

# -------------------------------------------------
# <editor-fold desc="CHECK FOR SOLUTIONS TO THE STRIPES PROBLEM">
amount = 0
f11 = plt.figure(11)
ax = f11.add_subplot(111)
args = [ax]
out = None

def shift(amount, ax):
    ax.clear()
    data = np.copy(avg_lfps_around_event)
    data[np.arange(0, 72, 2), :] += amount
    data = np.flipud(binning.rolling_window_with_step(data, np.mean, smooth_factor, smooth_factor))
    ax.imshow(data, aspect='auto', interpolation=interpolation)
    return None

sl.connect_repl_var(globals(), 'amount', 'out', 'shift', 'args', slider_limits=[-100, 100])


dataset_fliped = np.copy(dataset)
dataset_fliped[np.arange(0, 72, 2)] = -dataset_fliped[np.arange(0,72, 2)]
f11 = plt.figure(11)
plt.imshow(np.flipud(dataset_fliped), aspect='auto', extent=[-window_timepoints / const.SAMPLING_FREQUENCY,
                                                      window_timepoints / const.SAMPLING_FREQUENCY,
                                                      lfp_probe_positions[0], lfp_probe_positions[-2]],
           interpolation=interpolation)

#   Check if the AP data are stripy (they are)
ap_data_file_name = join(data_folder, 'Amplifier_APs.bin')
ap_data = ns_funcs.load_binary_amplifier_data(ap_data_file_name, const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

aps_indices_next_to_lfps = []
for i in np.arange(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE):
    aps_indices_next_to_lfps.append(8 + i * 20 - i)

times = np.arange(1000000, 89242000, 2000000)
ap_next_to_lfp = np.zeros((72, 16*const.SAMPLING_FREQUENCY))
for i in np.arange(len(times)):
    ap_next_to_lfp += np.abs(ap_data[aps_indices_next_to_lfps, times[i]:times[i] + 16*const.SAMPLING_FREQUENCY])

ap_next_to_lfp /= len(times)
ap_next_to_lfp_smooth = binning.rolling_window_with_step(ap_next_to_lfp, np.mean, 1000, 1000)

f = plt.figure(1)
plt.imshow(ap_next_to_lfp_smooth, aspect='auto', vmin=ap_next_to_lfp_smooth.min(), vmax=ap_next_to_lfp_smooth.max())

f = plt.figure(2)
plt.imshow(ap_next_to_lfp_smooth[np.arange(0, 72, 2), :], aspect='auto', vmin=ap_next_to_lfp_smooth.min(), vmax=ap_next_to_lfp_smooth.max())

f = plt.figure(3)
plt.imshow(ap_next_to_lfp_smooth[np.arange(1, 72, 2), :], aspect='auto', vmin=ap_next_to_lfp_smooth.min(), vmax=ap_next_to_lfp_smooth.max())

# </editor-fold>
# -------------------------------------------------

