

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis.Statistics import binning
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import sequence_viewer as sv
import transform as tr
import one_shot_viewer as osv

import pandas as pd
import matplotlib.pyplot as plt
import common_data_transforms as com_tr

from io import StringIO
import sys

# -------------------------------------------------
# LOAD FOLDERS
# -------------------------------------------------
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
denoised_data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                            'Denoised', 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')

events_folder = join(data_folder, "events")

time_points_buffer = 5200

ap_data = ns_funcs.load_binary_amplifier_data(join(data_folder, 'Amplifier_APs.bin'),
                                              number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

ap_data_panes = np.swapaxes(np.reshape(ap_data, (ap_data.shape[0],
                                                 int(ap_data.shape[1] / time_points_buffer), time_points_buffer)),
                            0, 1)

ap_den_data = ns_funcs.load_binary_amplifier_data(join(denoised_data_folder, 'Amplifier_APs_Denoised.bin'),
                                                  number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)
ap_den_data_panes = np.swapaxes(np.reshape(ap_den_data, (ap_den_data.shape[0],
                                                         int(ap_den_data.shape[1] / time_points_buffer),
                                                         time_points_buffer)),
                                0, 1)


# -------------------------------------------------
# QUICK LOOK AT NEURONS FIRING
# -------------------------------------------------
pane = 120
colormap = 'jet'
image_levels = [0, 150]
sv.image_sequence(globals(), 'pane', 'ap_data_panes', image_levels=image_levels, colormap=colormap, flip='ud')
sv.image_sequence(globals(), 'pane', 'ap_den_data_panes', image_levels=image_levels, colormap=colormap, flip='ud')


# -------------------------------------------------
#  CREATING NEURON FIRING RATES
# -------------------------------------------------
# Creating a scatter plot for neurons spiking
spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))
spike_info = pd.read_pickle(join(spikes_folder, 'spike_info.df'))
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

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
tr.connect_repl_var(globals(), 'pane', 'raster', 'raster_from_pane')


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


# -------------------------------------------------
# CHECKING SIMILARITIES AND DIFFERENCES BETWEEN THE TWO MODES OF FIRING NEURONS (SLOW AND FAST)
# -------------------------------------------------

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


plt.hist(template_info['firing rate'], bins=np.logspace(np.log10(0.001), np.log10(100), 50))
plt.gca().set_xscale("log")


rate_type_cutoff = 0.05  # = 3 times a minute
fast_neurons = template_info[template_info['firing rate'] >= rate_type_cutoff]['template number']
slow_neurons = template_info[template_info['firing rate'] < rate_type_cutoff]['template number']

avg_templates = np.load(join(spikes_folder, 'avg_spike_template.npy'))
avg_template_indices_for_fast_neurons = template_info[np.isin(template_info['template number'], fast_neurons.values)].\
    index.values
avg_template_indices_for_slow_neurons = template_info[np.isin(template_info['template number'], slow_neurons.values)].\
    index.values

# Find fast neurons
fast_neurons_biggest_channels = []
fast_indices = []
for neuron in avg_template_indices_for_fast_neurons:
    channel = np.squeeze(np.argwhere(avg_templates[neuron, :, :] == np.nanmin(avg_templates[neuron, :, :])))[0]
    if channel.__class__ is np.ndarray:
        channel = channel[0]
    fast_neurons_biggest_channels.append(avg_templates[neuron, channel, :])
    fast_indices.append(neuron)

fast_neurons_biggest_channels = np.array(fast_neurons_biggest_channels)

# and plot them
tim = np.arange(-30/const.SAMPLING_FREQUENCY, 30/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY)
f_fast = plt.figure(0)
ax_fast = f_fast.add_subplot(111)
ax_fast.plot(tim, fast_neurons_biggest_channels.T)
ax_fast.plot(tim, np.mean(fast_neurons_biggest_channels, axis=0), c=(0,0,0), linewidth=5)


# There are a good few slow neurons that have their peak really offset in time from the middle time point.
# That is because these 'neurons' have double peaks
# The following code REMOVES those (not useful)
'''
slow_neurons_biggest_channels = []
clean_slow_indices = []
for neuron in avg_template_indices_for_slow_neurons:
    channel_time = np.squeeze(np.argwhere(avg_templates[neuron, :, :] == np.nanmin(avg_templates[neuron, :, :])))
    if len(channel_time.shape) > 1:
        channel_time = channel_time[0]
    if channel_time[1] < 25 or channel_time[1] > 35:
        pass
    else:
        channel = channel_time[0]
        slow_neurons_biggest_channels.append(avg_templates[neuron, channel, :])
        clean_slow_indices.append(neuron)

slow_neurons_biggest_channels = np.array(slow_neurons_biggest_channels)
clean_slow_indices = np.array(clean_slow_indices)

centered_avg_template_indices = np.concatenate((avg_template_indices_for_fast_neurons, clean_slow_indices))
firing_rates_of_centered_neurons = template_info.iloc[centered_avg_template_indices]['firing rate'].values

pf.plot_log_histogram(firing_rates_of_centered_neurons, 50, 0.001, 50)
'''

# Here I SHIFT the double spiking neurons to the middle time point
slow_neurons_biggest_channels = []
single_slow_indices = []
double_slow_indices = []
for neuron in avg_template_indices_for_slow_neurons:
    channel_time = np.squeeze(np.argwhere(avg_templates[neuron, :, :] == np.nanmin(avg_templates[neuron, :, :])))
    if len(channel_time.shape) > 1:
        channel_time = channel_time[0]
    if channel_time[1] < 25:

        min_time = channel_time[1]
        data = avg_templates[neuron, channel_time[0], :]
        data_new = np.zeros(60)
        data_new[30-min_time:30+min_time] = data[:2 * min_time]
        slow_neurons_biggest_channels.append(data_new)
        double_slow_indices.append(neuron)
    elif channel_time[1] > 35:

        min_time = channel_time[1]
        data = avg_templates[neuron, channel_time[0], :]
        data_new = np.zeros(60)
        data_new[30 - (60 - min_time):30 + (60-min_time)] = data[-2*(60 - min_time):]
        slow_neurons_biggest_channels.append(data_new)
        double_slow_indices.append(neuron)
    else:
        channel = channel_time[0]
        slow_neurons_biggest_channels.append(avg_templates[neuron, channel, :])
        single_slow_indices.append(neuron)

slow_neurons_biggest_channels = np.array(slow_neurons_biggest_channels)

# Plot slow neurons
f_slow = plt.figure(1)
ax_slow = f_slow.add_subplot(111)
ax_slow.plot(tim, slow_neurons_biggest_channels.T)
ax_slow.plot(tim, np.mean(slow_neurons_biggest_channels, axis=0), c=(0,0,0), linewidth=5)


spp.view_grouped_templates_positions(spikes_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info.iloc[avg_template_indices_for_fast_neurons])
spp.view_grouped_templates_positions(spikes_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info.iloc[avg_template_indices_for_slow_neurons])


# CHECK IF THE DOUBLE FIRING NEURONS ARE ACTUALLY EXISTING SINGLE FIRING NEURONS THAT KILOSORT SAW AS DIFFERENT
# WHEN THEY FIRED VERY FAST
template_info_double_slow = template_info.iloc[double_slow_indices]
template_info_fast = template_info.iloc[fast_indices]
template_info_fast_and_double_slow = pd.concat([template_info_fast, template_info_double_slow])

spp.view_grouped_templates_positions(spikes_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info_double_slow)

spp.view_grouped_templates_positions(spikes_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info_fast_and_double_slow)

# ---------------------------
# USE THE FOLLOWING TO INTERACTIVELY SEE THE TEMPLATE TIME PLOT WHEN CLICKING ON A POINT ON A TEMPLATE POSITIONS PLOT
# ---------------------------

f = plt.figure(0)
old_stdout = sys.stdout
global previous_template_number
previous_template_number = -1
global result
result = StringIO()
template_number = 0


def show_average_template(figure):
    global previous_template_number
    global result
    sys.stdout = result
    string = result.getvalue()
    new = string[-200:]
    try:
        template_number = int(new[new.find('Template number'): new.find('Template number')+22][18:22])
        if template_number != previous_template_number:
            template = template_info[template_info['template number'] == template_number]
            figure.clear()
            ax = figure.add_subplot(111)
            try:
                ax.plot(np.squeeze(avg_templates[template.index.values]).T)
            except:
                pass
        previous_template_number = template_number
        figure.suptitle('Template = {}, with {} number of spikes'.format(str(template_number),
                                                                         str(template['number of spikes'].values[0])))
    except:
        template_number = None
    return template_number


tr.connect_repl_var(globals(), 'f', 'template_number', 'show_average_template')
# ---------------------------
