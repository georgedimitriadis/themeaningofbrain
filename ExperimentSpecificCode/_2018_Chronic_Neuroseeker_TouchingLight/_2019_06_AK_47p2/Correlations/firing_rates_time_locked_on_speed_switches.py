

from os.path import join
import numpy as np
from BrainDataAnalysis import binning
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const

import sequence_viewer as sv
import drop_down as dd
import slider as sl

import matplotlib.pyplot as plt
import pandas as pd

#  -------------------------------------------------
# <editor-fold desc="GET FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

kilosort_folder = join(analysis_folder, 'Kilosort')
events_folder = join(data_folder, 'events')
dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')

markers_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1Jun30shuffle1_325000.h5')
labeled_video_file = join(dlc_project_folder, 'videos',
                          r'Croped_videoDeepCut_resnet50_V1Jun30shuffle1_325000_labeled.mp4')
crop_window_position_file = join(dlc_folder, 'BonsaiCroping', 'Croped_Video_Coords.csv')
full_video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))
video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
# </editor-fold>

#  -------------------------------------------------
# <editor-fold desc="CREATE SPEEDS AND CHECK THEM ON THE VIDEO">
# Use body position to create velocities (both linear and polar)
conversion_const = const.PIXEL_PER_FRAME_TO_CM_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                                 180 * (1/np.pi) * np.arctan2(body_velocities[:, 1], body_velocities[:, 0])]).transpose()

# Correct the speeds by removing the very large spikes (replace them with the average speeds around them)
speeds = body_velocities_polar[:, 0].copy()
acc = np.diff(speeds)
for i in np.arange(len(acc)):
    if speeds[i] > 200:
        speeds[i] = np.mean(speeds[i-10:i+10])


num_of_frames_to_average = 1/(1/120)
speeds_smoothed = binning.rolling_window_with_step(speeds, np.mean,
                                               num_of_frames_to_average, 1)
speeds_smoothed[0] = 0

speeds_smoothed = np.array(speeds_smoothed)
rest_speed_threshold_cm_per_sec = 15
rest_move_periods = np.zeros(len(speeds_smoothed))
rest_move_periods[np.where(speeds_smoothed > rest_speed_threshold_cm_per_sec)] = 1

# plt.plot(rest_move_periods)
# plt.plot((speeds_smoothed - np.nanmin(speeds_smoothed)) / np.nanmax(speeds_smoothed - np.nanmin(speeds_smoothed)))

# Check the calculated speeds on the video
frame = 1
sv.image_sequence(globals(), 'frame', 'full_video_file')

# </editor-fold>

#  -------------------------------------------------
# <editor-fold desc="FIND THE FRAMES WHERE THE RAT SWITCHES MOVEMENT STATE">
seconds_of_constant_state = 5
switches_from_rest_to_move = np.squeeze(np.argwhere(np.diff(rest_move_periods) > 0))
switches_from_move_to_rest = np.squeeze(np.argwhere(np.diff(rest_move_periods) < 0))
switches_from_long_rest_to_long_move = []
for r_m in switches_from_rest_to_move[1:]:
    if np.all(rest_move_periods[r_m - seconds_of_constant_state * 120 : r_m - 1] == 0) and \
       np.all(rest_move_periods[r_m + 1: r_m + seconds_of_constant_state * 120] == 1):
        switches_from_long_rest_to_long_move.append(r_m + 70)

switches_from_long_rest_to_long_move = np.array(switches_from_long_rest_to_long_move)

switches_from_long_move_to_long_rest = []
for m_r in switches_from_move_to_rest[1:]:
    if np.all(rest_move_periods[m_r - seconds_of_constant_state * 120 : m_r - 1] == 1) and \
       np.all(rest_move_periods[m_r + 1: m_r + seconds_of_constant_state * 120] == 0):
        switches_from_long_move_to_long_rest.append(m_r + 70)

switches_from_long_move_to_long_rest = np.array(switches_from_long_move_to_long_rest)

def nothing(f):
    return f


dd.connect_repl_var(globals(), 'switches_from_long_move_to_long_rest', 'frame')
# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="AVERAGE FIRING RATE AROUND MOMENTS OF SWITCHING MOVEMENT STATE">
time_around_switch = 5
switches = switches_from_long_rest_to_long_move

frames_around_switch = time_around_switch * 120
time_points_around_switch = time_around_switch * const.SAMPLING_FREQUENCY
base_end = int(frames_around_switch * 0.7)
switch_start = int(frames_around_switch * 1.3)
switch_end = int(frames_around_switch * 2)


firing_rate_around_switches = np.zeros((len(switches), len(spike_rates), 2 * frames_around_switch))

for f in np.arange(len(switches)):
    frame = switches[f]
    firing_rate_around_switches[f, :, :] = spike_rates[:, frame - frames_around_switch:
                                                            frame + frames_around_switch]

avg_firing_rate_around_switches = firing_rate_around_switches.mean(axis=0)

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="GET THE NEURONS WITH AN AVERAGE LARGE INCREASE AFTER THE EVENT (EITHER STOP TO START OR START TO STOP)">
increasing_firing_rates = []
increasing_firing_rates_neuron_index = []
increasing_firing_rates_ratio = []
for n in np.arange(len(avg_firing_rate_around_switches)):
    neuron = avg_firing_rate_around_switches[n]
    if neuron[:base_end].mean() > 0.1:
        if neuron[:base_end].mean() * 3 < \
                neuron[switch_start:switch_end].mean():
            increasing_firing_rates.append(neuron)
            increasing_firing_rates_neuron_index.append(n)
            increasing_firing_rates_ratio.append(neuron[switch_start:switch_end].mean()
                                                 / neuron[:base_end].mean())
increasing_firing_rates = np.array(increasing_firing_rates)
increasing_firing_rates_neuron_index = np.array(increasing_firing_rates_neuron_index)
increasing_firing_rates_ratio = np.array(increasing_firing_rates_ratio)

print(increasing_firing_rates.shape)

# Show all the increasing f.r. neurons
fig = plt.figure(0)
fig.set_figheight(2)
fig.set_figwidth(10)
ax = fig.add_subplot(111)
ax.vlines(x=0, ymin=0, ymax=len(increasing_firing_rates))

ax.imshow(increasing_firing_rates, vmax=increasing_firing_rates.max(), vmin=0, extent=[-frames_around_switch * 8.3,
                                                                                       frames_around_switch * 8.3,
                                                                                       0,
                                                                                       len(increasing_firing_rates)])
ax.set_aspect(200)


# Have a detailed look at the neuron with the largest increase
# largest_increase_neuron_index = increasing_firing_rates_neuron_index[np.argmax(increasing_firing_rates_ratio)]
index = 0
fig1 = plt.figure(0)
fig2 = plt.figure(1)
output = None
args = [fig1, fig2]


def show_rasters_increase(index, fig1, fig2):
    largest_increase_neuron_index = increasing_firing_rates_neuron_index[index]
    largest_increase_firing_rate = avg_firing_rate_around_switches[largest_increase_neuron_index]

    fig1.clear()
    ax1 = fig1.add_subplot(111)
    ax1.plot(largest_increase_firing_rate)
    ax1.vlines(x=frames_around_switch, ymin=largest_increase_firing_rate.min(),
               ymax=largest_increase_firing_rate.max())

    largest_increase_neurons_spikes = template_info.iloc[largest_increase_neuron_index]['spikes in template']
    largest_increase_neurons_spike_times = spike_info.loc[np.isin(spike_info['original_index'],
                                                                  largest_increase_neurons_spikes)]['times'].values.\
        astype(np.int64)

    largest_increase_neuron_raster = []
    for trial in switches:
        largest_increase_neuron_raster.append((largest_increase_neurons_spike_times - trial * 8.33) / const.SAMPLING_FREQUENCY)

    largest_increase_neuron_raster = np.array(largest_increase_neuron_raster)

    fig2.clear()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(largest_increase_neuron_raster, np.tile(np.arange(len(largest_increase_neuron_raster)),
                                                        largest_increase_neuron_raster.shape[1]),
                s=10)
    ax2.set_xlim(-time_points_around_switch / const.SAMPLING_FREQUENCY,
                 time_points_around_switch / const.SAMPLING_FREQUENCY)

    return None


sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_increase', 'args',
                    slider_limits=[0, len(increasing_firing_rates_neuron_index) - 1])
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="GET THE NEURONS WITH AN AVERAGE LARGE DECREASE AFTER THE EVENT (EITHER STOP TO START OR START TO STOP)">
decreasing_firing_rates = []
decreasing_firing_rates_neuron_index = []
decreasing_firing_rates_ratio = []
for n in np.arange(len(avg_firing_rate_around_switches)):
    neuron = avg_firing_rate_around_switches[n]
    if neuron[:base_end].mean() > 0.5:
        if neuron[:base_end].mean() > \
                neuron[switch_start:switch_end].mean() * 3:
            decreasing_firing_rates.append(neuron)
            decreasing_firing_rates_neuron_index.append(n)
            decreasing_firing_rates_ratio.append(neuron[:base_end].mean() -
                                                 neuron[switch_start:switch_end].mean())
decreasing_firing_rates = np.array(decreasing_firing_rates)
decreasing_firing_rates_neuron_index = np.array(decreasing_firing_rates_neuron_index)
decreasing_firing_rates_ratio = np.array(decreasing_firing_rates_ratio)
print(decreasing_firing_rates.shape)

# Show all the decreasing f.r. neurons
fig = plt.figure(1)
fig.set_figheight(4)
fig.set_figwidth(10)
ax = fig.add_subplot(111)
ax.vlines(x=0, ymin=0, ymax=len(decreasing_firing_rates))
ax.imshow(decreasing_firing_rates, vmax=decreasing_firing_rates.max(), vmin=0, extent=[-frames_around_switch*8.3,
                                                                                       frames_around_switch * 8.3,
                                                                                       0,
                                                                                       len(decreasing_firing_rates)])
ax.set_aspect(200)

# </editor-fold>