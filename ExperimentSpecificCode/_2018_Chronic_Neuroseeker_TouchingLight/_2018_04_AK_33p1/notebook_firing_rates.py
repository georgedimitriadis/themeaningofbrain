

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from BrainDataAnalysis import binning

import sequence_viewer as sv
import transform as tr
import one_shot_viewer as osv

import pandas as pd
import matplotlib.pyplot as plt
import common_data_transforms as com_tr


date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
denoised_data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                            'Denoised', 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')

events_folder = join(data_folder, "events")

time_points_buffer = 5200

lfp_data = ns_funcs.load_binary_amplifier_data(join(data_folder, 'Amplifier_LFPs.bin'),
                                               number_of_channels=const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

lfp_data_panes = np.swapaxes(np.reshape(lfp_data, (lfp_data.shape[0], int(lfp_data.shape[1] / time_points_buffer), time_points_buffer)), 0, 1)

ap_data = ns_funcs.load_binary_amplifier_data(join(data_folder, 'Amplifier_APs.bin'),
                                              number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

ap_data_panes = np.swapaxes(np.reshape(ap_data, (ap_data.shape[0], int(ap_data.shape[1] / time_points_buffer), time_points_buffer)), 0, 1)

ap_den_data = ns_funcs.load_binary_amplifier_data(join(denoised_data_folder, 'Amplifier_APs_Denoised.bin'),
                                              number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)
ap_den_data_panes = np.swapaxes(np.reshape(ap_den_data, (ap_den_data.shape[0], int(ap_den_data.shape[1] / time_points_buffer),
                                                     time_points_buffer)), 0, 1)



pane = 120
colormap = 'jet'
image_levels = [0, 150]
sv.image_sequence(globals(), 'pane', 'ap_data_panes', image_levels=image_levels, colormap=colormap, flip='ud')
sv.image_sequence(globals(), 'pane', 'ap_den_data_panes', image_levels=image_levels, colormap=colormap, flip='ud')

lfp_channels_on_probe = np.arange(9, 1440, 20)
channels_heights = ns_funcs.get_channels_heights_for_spread_calulation(lfp_channels_on_probe)
bad_lfp_channels = [35, 36, 37]
lfp_channels_used = np.delete(np.arange(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE), bad_lfp_channels)


def spread_lfp_pane(p):
    pane = lfp_data_panes[p, :, :]
    spread = ns_funcs.spread_data(pane, channels_heights, lfp_channels_used)
    spread = np.flipud(spread)
    return spread


pane_data = None
tr.connect_repl_var(globals(), 'pane', 'spread_lfp_pane', 'pane_data')

osv.graph(globals(), 'pane_data')


camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
points_per_pulse = np.mean(np.diff(camera_pulses))

camera_frames_in_video = csv_funcs.get_true_frame_array(data_folder)
time_point_of_first_video_frame = camera_pulses[camera_frames_in_video][0]

video_frame = 0
video_file = join(data_folder, 'Video.avi')
sv.image_sequence(globals(), 'video_frame', 'video_file')


def pane_to_frame(x):
    time_point = (x + 0.4) * time_points_buffer
    return sync_funcs.time_point_to_frame(time_point_of_first_video_frame, camera_frames_in_video,
                                                               points_per_pulse, time_point)


tr.connect_repl_var(globals(), 'pane', 'pane_to_frame', 'video_frame')


#  CREATING NEURON FIRING RATES
#  Separating neurons to region and having a look at their distributions

spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))


brain_regions = const.BRAIN_REGIONS
cortex = np.array([brain_regions['Cortex MPA'], brain_regions['CA1']]) / const.POSITION_MULT
hippocampus = np.array([brain_regions['CA1'], brain_regions['Thalamus LPMR']]) / const.POSITION_MULT
thalamus = np.array([brain_regions['Thalamus LPMR'], brain_regions['Zona Incerta']]) / const.POSITION_MULT
sub_thalamic = np.array([brain_regions['Zona Incerta'], 0]) / const.POSITION_MULT

cort_cells = template_info[np.logical_and(template_info['position Y'] < cortex[0], template_info['position Y'] > cortex[1])]
hipp_cells = template_info[np.logical_and(template_info['position Y'] < hippocampus[0], template_info['position Y'] > hippocampus[1])]
thal_cells = template_info[np.logical_and(template_info['position Y'] < thalamus[0], template_info['position Y'] > thalamus[1])]
sub_th_cells = template_info[np.logical_and(template_info['position Y'] < sub_thalamic[0], template_info['position Y'] > sub_thalamic[1])]


plt.hist(sub_th_cells['firing rate'], bins=np.logspace(np.log10(0.001), np.log10(100), 50))
plt.gca().set_xscale("log")


# Creating a scatter plot for neurons spiking
fast_neuron = template_info[template_info['firing rate'] == template_info['firing rate'].max()]['template number']
spike_times = spike_info[spike_info['template_after_sorting'] == fast_neuron.values[0]]['times'].values

global x
global y

def raster_from_pane_one_neuron(p):
    global x
    global y
    start = p*time_points_buffer
    end = (p+1)*time_points_buffer
    x = [start]
    times = spike_times[np.logical_and(spike_times < end, spike_times > start)]
    for spike in times:
        x.append(spike)
    x.append(end)
    y = np.ones(len(x))
    y[0] = 0
    y[-1] = 0
    return x, y

def raster_from_pane(p):
    global x
    global y
    start = p*time_points_buffer
    end = (p+1)*time_points_buffer

    x = spike_info['times'].values[np.logical_and(spike_info['times'] < end, spike_info['times'] > start)]
    templates = spike_info['template_after_sorting'].values[np.logical_and(spike_info['times'] < end, spike_info['times'] > start)]
    y = []
    for t in templates:
        y.append(template_info[template_info['template number'] == t]['position Y'].values[0])
    y = np.array(y) * const.POSITION_MULT
    return x, y

raster = None
tr.connect_repl_var(globals(), 'pane', 'raster_from_pane', 'raster')


osv.graph(globals(), 'y', 'x', True)

# Calculating firing rates using arbitrary time windows
'''
seconds_in_averaging_window = 0.5
averaging_window = int(seconds_in_averaging_window * const.SAMPLING_FREQUENCY)
num_of_windows = int(ap_data.shape[1] / averaging_window)
spike_rates = np.zeros((len(template_info), num_of_windows))


for t_index in np.arange(len(template_info)):
    template_index = template_info['template number'].iloc[t_index]
    spike_times_in_template = spike_info[spike_info['template_after_sorting'] == template_index]['times'].values

    for s_index in np.arange(num_of_windows):
        start = s_index * averaging_window
        end = start + averaging_window
        spikes_in_window = spike_times_in_template[np.logical_and(spike_times_in_template < end, spike_times_in_template > start)]
        spike_rates[t_index, s_index] = len(spikes_in_window) / seconds_in_averaging_window

np.save(join(spikes_folder, 'firing_rate_with_0p50s_window.npy'), spike_rates)
'''

# Make the spike rates using each frame as a binning window
'''
#  Load the pre generated DataFrames for the event CSVs
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(spikes_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))
sampling_frequency = const.SAMPLING_FREQUENCY
spike_rates = binning.spike_count_per_frame(template_info, spike_info, event_dataframes['ev_video'],
                                            sampling_frequency, file_to_save_to=file_to_save_to)

# Using the frame based spikes rates do a rolling window to average a bit more
num_of_frames_to_average = 0.25/(1/120)

spike_rates_0p25 = []
for n in np.arange(spike_rates.shape[0]):
    spike_rates_0p25.append(binning.rolling_window_with_step(spike_rates[n, :], np.mean,
                                                        num_of_frames_to_average, num_of_frames_to_average))
spike_rates_0p25 = np.array(spike_rates_0p25)
np.save(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'), spike_rates_0p25)
'''

# Have a look
spike_rates_0p25 = np.load(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'))
spike_rates_fixed = np.copy(spike_rates_0p25)
spike_rates_fixed[601, :5] = 0

image_levels_sr = [0, 200]
cm_sr = 'jet'

osv.image(globals(), 'spike_rates_fixed', 'image_levels_sr', 'cm_sr')

spike_rates_spaced = com_tr.space_data(spike_rates_0p25, 100)




