

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import time

import one_shot_viewer as osv
import sequence_viewer as sv
import drop_down as dd
import transform as tr
import slider as sl

from ExperimentSpecificCode._2019_EI_Anesthesia import constants as const
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs


# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
# ----------------------------------------------------------------------------------------------------------------------
analysis_folder = r'F:\Neuroseeker_EI\2019_05_06\Analysis'
brain_data_folder = r'F:\Neuroseeker_EI\2019_05_06\Data\NeuroSeeker'
kilosort_folder = join(analysis_folder, 'Kilosort')
events_folder = r'F:\Neuroseeker_EI\2019_05_06\Analysis\Events'

events_dataframe_filename = join(events_folder, 'events.df')
template_info_filename = join(kilosort_folder, 'template_info.df')
spike_info_filename = join(kilosort_folder, 'spike_info_after_cleaning.df')

binary_data_filename = join(brain_data_folder, r'concatenated_data_before_and_after_muscimol_APs.bin')

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------------------------------------------------

events = pd.read_pickle(events_dataframe_filename)

template_info = np.load(template_info_filename, allow_pickle=True)
spike_info = np.load(spike_info_filename, allow_pickle=True)

ap_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                              number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

muscimol_injection_times = np.load(join(events_folder, 'muscimol_injection_times.npy'))
# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

length_of_six_pip_sound = const.LENGTH_OF_SIX_PIP_SOUND
length_of_one_pip = const.LENGTH_OF_ONE_PIP
time_between_pips = const.TIME_BETWEEN_PIPS
inter_sound_interval = const.INTER_SOUND_INTERVAL
sampling_freq = const.SAMPLING_FREQUENCY
end_time_of_no_muscimol = muscimol_injection_times[0][0]  # 43748034
start_time_of_muscimol = muscimol_injection_times[0][-1]

cutoff_firing_rate = 1
fast_neurons = template_info[template_info['firing rate'] > cutoff_firing_rate]
# ----------------------------------------------------------------------------------------------------------------------

# Calculating firing rates using arbitrary time windows
'''
seconds_in_averaging_window = 0.2
averaging_window = int(seconds_in_averaging_window * const.SAMPLING_FREQUENCY)
num_of_windows = int(ap_data.shape[1] / averaging_window)
spike_rates = np.zeros((len(template_info), num_of_windows))


for t_index in np.arange(len(template_info)):
    template_index = template_info['template number'].iloc[t_index]
    spike_times_in_template = spike_info[spike_info['template_after_sorting'] == template_index]['times'].values

    for s_index in np.arange(num_of_windows):
        start = s_index * averaging_window
        end = start + averaging_window
        spikes_in_window = spike_times_in_template[np.logical_and(spike_times_in_template < end, 
                                                                  spike_times_in_template > start)]
        spike_rates[t_index, s_index] = len(spikes_in_window) / seconds_in_averaging_window

np.save(join(kilosort_folder, 'firing_rate_with_0p2s_window.npy'), spike_rates)
'''


# Calculate PSTHs
def make_tuning_curves_of_neurons(pdf_filename, template_info, spike_info,
                                  firing_rate_cutoff=1, time_pre=0.25, time_post=0.20,
                                  trial_duration=0.7, binwidth=0.01):

    n_pre = int(round(time_pre / binwidth + .5))
    n_post = int(round(time_post / binwidth + .5))
    edges = np.arange(-n_pre, np.round(trial_duration / binwidth + .5)+n_post) * binwidth

    fast_neurons = template_info[template_info['firing rate'] > firing_rate_cutoff]

    def get_spike_times_of_fast_neuron_index(neuron_index):
        neuron_template = fast_neurons['template number'].iloc[neuron_index]
        spikes = template_info['spikes in template'].loc[template_info['template number'] == neuron_template].values[0]
        spike_times = spike_info['times'].loc[np.isin(spike_info['original_index'], spikes)].values

        return spike_times

    def get_event_times_for_frequency(freq):
        times = events['time_points'].loc[events['frequencies'] == freq].values
        times_no_muscimol = times[times < end_time_of_no_muscimol]

        return times_no_muscimol

    def make_hists_for_neuron(fast_neuron_index, fig):
        fig.clear()
        plot_index = 1
        avg_hists_freq = {}
        for freq in np.arange(5, 16):
            ax = fig.add_subplot(3, 4, plot_index)
            ax.set_title('Freq = {}'.format(str(freq)))
            plot_index += 1
            avg_hist = []
            times_no_muscimol = get_event_times_for_frequency(freq)
            for event_time in times_no_muscimol:
                spike_times = get_spike_times_of_fast_neuron_index(fast_neuron_index)
                times_to_hist = (spike_times - event_time) / sampling_freq
                count, _ = np.histogram(times_to_hist, bins=edges)
                avg_hist.append(count)

            avg_hist = np.array(avg_hist)
            avg_hist = np.mean(avg_hist, axis=0)

            _ = ax.bar(edges[:-1]*1000,
                       avg_hist / binwidth,
                       color='#1f77b4',
                       width=binwidth*1000,
                       ec='none')
            avg_hists_freq[freq] = avg_hist
        return avg_hists_freq

    pdf = PdfPages(pdf_filename)
    avg_hists = dict()
    for fast_neuron_index in np.arange(len(fast_neurons)):
        neuron_template_name = str(fast_neurons['template number'].iloc[fast_neuron_index])
        print('Making neuron template {}, index {} of {}'.format (neuron_template_name, str(fast_neuron_index),
                                                                  str(len(fast_neurons))))
        fig = plt.figure(0)
        fig.suptitle('Neuron template = {}'.format(neuron_template_name), fontsize=20)

        avg_hists_freq = make_hists_for_neuron(fast_neuron_index, fig)
        avg_hists[fast_neurons['template number'].iloc[fast_neuron_index]] = avg_hists_freq
        pdf.savefig(fig)
        plt.close()
    avg_hists_name = join(analysis_folder, 'Results', 'all_neurons_larger_than_{}Hz_tuning_curves.pkl'.
                                                       format(str(firing_rate_cutoff)))
    pickle.dump(avg_hists, open(avg_hists_name, 'wb'))
    pdf.close()
    return avg_hists


firing_rate_cutoff = 1
pdf_filename = join(analysis_folder, 'Results', r'Tuning curves of neurons with bigger than {} Hz firing rate 2.pdf'.
                    format(str(firing_rate_cutoff)))

avg_hists = make_tuning_curves_of_neurons(pdf_filename, template_info, spike_info,
                                          firing_rate_cutoff=firing_rate_cutoff, time_pre=0.25, time_post=0.20,
                                          trial_duration=length_of_six_pip_sound/sampling_freq, binwidth=0.01)


avg_hists = np.load(join(analysis_folder, 'Results', 'all_neurons_larger_than_{}Hz_tuning_curves.npy'.
                 format(str(firing_rate_cutoff))), allow_pickle=True)

# ----------------------------------------------------------------------------------------------------------------------

# Calculate PSTHs more detailed

time_pre=0.25
time_post=0.20
trial_duration=0.7
binwidth=0.025

n_pre = int(round(time_pre / binwidth + .5))
n_post = int(round(time_post / binwidth + .5))
edges = np.arange(-n_pre, np.round(trial_duration / binwidth + .5)+n_post) * binwidth

# This is to speed up the search for spike times in get_spike_times_of_fast_neuron_index()
all_spike_time_with_original_index = np.zeros(spike_info['original_index'].values.max() + 1)
all_spike_time_with_original_index[spike_info['original_index'].values] = spike_info['times'].values


def get_spike_times_of_fast_neuron_index(neuron_index):
    neuron_template = fast_neurons['template number'].iloc[neuron_index]
    spikes = template_info['spikes in template'].loc[template_info['template number'] == neuron_template].values[0]
    spike_times = all_spike_time_with_original_index[spikes]

    return spike_times


def get_event_times_for_frequency_intensity_pair(freq, nominal_intensity, pre_muscimol=True):
    event_times = events['time_points'].loc[np.logical_and(events['frequencies'] == freq,
                                                     events['nominal_intensities'] == nominal_intensity)].values
    if pre_muscimol:
        event_times = event_times[event_times < end_time_of_no_muscimol]
    else:
        event_times = event_times[event_times > start_time_of_muscimol]

    return event_times


def get_hist_for_neuron_freq_intensity_set(neuron_index, figure, freq, nominal_intensity, pre_muscimol=True):

    event_times = get_event_times_for_frequency_intensity_pair(freq, nominal_intensity, pre_muscimol=pre_muscimol)

    avg_hist = []
    for event_time in event_times:
        spike_times = get_spike_times_of_fast_neuron_index(neuron_index)
        times_to_hist = (spike_times - event_time) / sampling_freq
        count, _ = np.histogram(times_to_hist, bins=edges)

        avg_hist.append(count)

    avg_hist = np.array(avg_hist)
    avg_hist = np.mean(avg_hist, axis=0)

    figure.clear()
    ax = figure.add_subplot(111)
    neuron_template = fast_neurons['template number'].iloc[neuron_index]
    ax.set_title('Neuron Template = {}, Freq = {}, Intensity = {}'.
                 format(str(neuron_template),str(freq), str(nominal_intensity)))

    _ = ax.bar(edges[:-1] * 1000,
               avg_hist / binwidth,
               color='#1f77b4',
               width=binwidth * 1000,
               ec='none')


neuron_index = 0
out = None
figure = plt.figure(0)
freq = 5
nominal_intensity = 0
pre_muscimol = True

args = [figure, freq, nominal_intensity, pre_muscimol]


def change_freq(freq):
    args = [figure, freq, nominal_intensity, pre_muscimol]
    return args


def change_nominal_intensity(nominal_intensity):
    args = [figure, freq, nominal_intensity, pre_muscimol]
    return args


sl.connect_repl_var(globals(), 'freq', 'change_freq', 'args', slider_limits=[5, 15])
sl.connect_repl_var(globals(), 'nominal_intensity', 'change_nominal_intensity', 'args', slider_limits=[0, 3])
sl.connect_repl_var(globals(), 'neuron_index', 'get_hist_for_neuron_freq_intensity_set', 'out', 'args',
                    slider_limits=[0, 141])
