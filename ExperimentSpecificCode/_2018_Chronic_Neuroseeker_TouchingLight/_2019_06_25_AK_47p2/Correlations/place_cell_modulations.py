
from os.path import join
import numpy as np

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

import slider as sl


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

shuffled_filenames = {'dtp': 'shuffled_mut_info_spike_rate_960_vs_distance_to_poke.npy',
                      'speed': 'shuffled_mut_info_spike_rate_1140_vs_speed.npy'}
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

mi_spikes_vs_distance_to_poke = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke.npy'))
mi_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected.npy'))

mis = {'speed': mi_spikes_vs_speed,
       'dtp': mi_spikes_vs_distance_to_poke}

mis_shuffled = {}
for s in shuffled_filenames:
    mis_shuffled[s] = np.load(join(mutual_information_folder, shuffled_filenames[s]))

ti_decreasing_neurons_on_start_ballistic = np.load(join(ballistic_mov_folder, 'ti_decreasing_neurons_on_start_ballistic.npy'),
                                                   allow_pickle=True)
ti_increasing_neurons_on_start_ballistic = np.load(join(ballistic_mov_folder, 'ti_increasing_neurons_on_start_ballistic.npy'),
                                                   allow_pickle=True)
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
number_of_bins = 10

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="GET THE NEURONS WITH SIGNIFICANT MUTUAL INFORMATION CONTENT (FOR DTP AND SPEED)">

confidence_level = 0.99
correlated_neuron_indices = {}
mean_sh = {}
confi_intervals = {}
for s in mis_shuffled:
    mean_sh[s] = np.mean(mis_shuffled[s])
    confi_intervals[s] = mis_shuffled[s][int((1. - confidence_level) / 2 * 1000)], \
                      mis_shuffled[s][int((1. + confidence_level) / 2 * 1000)]

    correlated_neuron_indices[s] = np.squeeze(np.argwhere(mis[s] > mean_sh[s] + confi_intervals[s][1]))

#   Have a quick look
s = 'dtp'
plt.hist(mis[s], bins= 200, color=(0, 0, 1, 0.4))
plt.hist(mis_shuffled[s], bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh[s], mean_sh[s] + confi_intervals[s][0], mean_sh[s] + confi_intervals[s][1]], 0, 20)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="PICK THE HIPPOCAMPAL ONES">

correlated_neurons = {}
for s in mis:
    correlated_neurons[s] = template_info.loc[correlated_neuron_indices[s]]

hipp_borders = (3600, 4500)

hipp_correlated_neurons = {}
hipp_correlated_neurons_indices = {}
for s in mis:
    neurons = correlated_neurons[s]
    h = np.array(hipp_borders) / const.POSITION_MULT
    hipp_correlated_neurons[s] = neurons[np.logical_and(neurons['position Y'] > h[0], neurons['position Y'] < 2000)]
    hipp_correlated_neurons_indices[s] = hipp_correlated_neurons[s].index.values

#   Look at all the high MI neurons
s = 'speed'
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=correlated_neurons[s],
                                     dot_sizes=mis[s][correlated_neuron_indices[s]] * 4000,
                                     font_size=5)
#   Look at the hippocampal high MI neurons
s = 'dtp'
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=hipp_correlated_neurons[s],
                                     dot_sizes=mis[s][correlated_neuron_indices[s]] * 4000,
                                     font_size=5)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="CREATE THE OCCUPANCY MAPS TO LOOK FOR PLACE CELLS (RUN ONCE)">

bins = np.arange(0, 640 + 640/number_of_bins, 640/number_of_bins)

occupancy = []
for i_x in np.arange(len(bins) - 1):
    x = [bins[i_x], bins[i_x+1]]
    for i_y in np.arange(len(bins) - 1):
        y = [bins[i_y], bins[i_y + 1]]
        occupancy.append(len(np.argwhere(np.logical_and(np.logical_and(body_positions[:, 0] > x[0],
                                                                       body_positions[:, 0] < x[1]),
                                                        np.logical_and(body_positions[:, 1] > y[0],
                                                                       body_positions[:, 1] < y[1])))))

occupancy = np.array(occupancy)/len(body_positions)
occupancy_matrix = np.flipud(occupancy.reshape(number_of_bins, number_of_bins).T)

spike_probabilities_at_position = []
for neuron in np.arange(len(spike_rates)):
    fr = spike_rates[neuron]
    mean_fr = np.mean(fr)
    fr_at_position = []
    for i_x in np.arange(len(bins) - 1):
        x = [bins[i_x], bins[i_x+1]]
        for i_y in np.arange(len(bins) - 1):
            y = [bins[i_y], bins[i_y + 1]]
            frames = np.argwhere(np.logical_and(np.logical_and(body_positions[:, 0] > x[0],
                                                               body_positions[:, 0] <= x[1]),
                                                np.logical_and(body_positions[:, 1] > y[0],
                                                               body_positions[:, 1] <= y[1])))
            if len(frames) > 0:
                fr_at_position.append(np.mean(fr[frames]))
            else:
                fr_at_position.append(0)
    spike_probabilities_at_position.append(np.array(fr_at_position) / mean_fr)

spike_probabilities_at_position = np.array(spike_probabilities_at_position)
np.save(join(mutual_information_folder, 'spike_probabilities_at_position.npy'), spike_probabilities_at_position)

#   Find where poke is in the occupancy and spike probability maps
poke = [600, 320]
other_place = body_positions[393189]
poke_position = np.zeros(number_of_bins * number_of_bins)
for i_x in np.arange(len(bins) - 1):
    x = [bins[i_x], bins[i_x+1]]
    for i_y in np.arange(len(bins) - 1):
        y = [bins[i_y], bins[i_y + 1]]
        if poke[0] >= x[0] and poke[0] < x[1]:
                if poke[1] >= y[0] and poke[1] < y[1]:
                    poke_position[i_y*number_of_bins + i_x] = 1
        if other_place[0] >= x[0] and other_place[0] < x[1]:
                if other_place[1] >= y[0] and other_place[1] < y[1]:
                    poke_position[i_y*number_of_bins + i_x] = 1
poke_position = np.array(poke_position)
plt.figure(0).add_subplot(111).imshow(poke_position.reshape((number_of_bins, number_of_bins)))
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="FIND THE PLACE CELLS AND VISUALISE OCCUPANCY MAPS">

spike_probabilities_at_position = np.load(join(mutual_information_folder, 'spike_probabilities_at_position.npy'))

spike_probabilities_at_position_matrix = \
    np.zeros((spike_probabilities_at_position.shape[0], number_of_bins, number_of_bins))
for p in np.arange(spike_probabilities_at_position.shape[0]):
    prob = np.flipud(spike_probabilities_at_position[p].reshape(number_of_bins, number_of_bins).T)
    spike_probabilities_at_position_matrix[p, :, :] = prob

max_firing_rates_at_place = np.argwhere(spike_probabilities_at_position[hipp_correlated_neurons_indices['dtp']] > 3)
place_cells_index = hipp_correlated_neurons_indices['dtp'][np.unique(max_firing_rates_at_place[:, 0])]



def show_probs_only_dtp(neuron, f):
    f.clear()
    a = f.add_subplot(111)
    im = a.imshow(spike_probabilities_at_position_matrix[hipp_correlated_neurons_indices['dtp'][neuron]],
             interpolation='quadric', cmap='jet')
    plt.title('Neuron index {}'.format(hipp_correlated_neurons_indices['dtp'][neuron]))
    cax = f.add_axes([0.83, 0.1, 0.05, 0.78])
    f.colorbar(im, cax=cax,  orientation='vertical')
    return None


def show_probs(neuron, f):
    f.clear()
    a = f.add_subplot(111)
    im = a.imshow(spike_probabilities_at_position_matrix[neuron],
                  interpolation='quadric', cmap='jet')
    neuron_in_dtp = False
    if neuron in correlated_neuron_indices['dtp']:
        neuron_in_dtp = True
    plt.title('Neuron index {}. Is in DTP = {}'.format(str(neuron), str(neuron_in_dtp)))
    cax = f.add_axes([0.83, 0.1, 0.05, 0.78])
    f.colorbar(im, cax=cax, orientation='vertical')
    return None

n_index = 0
out = None
f = plt.figure(0)
args = [f]

sl.connect_repl_var(globals(), 'n_index', 'out', 'show_probs', 'args', slider_limits=[0, spike_rates.shape[0]])
sl.connect_repl_var(globals(), 'n_index', 'out', 'show_probs_only_dtp', 'args',
                    slider_limits=[0, len(hipp_correlated_neurons_indices['dtp'])])


# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="CHECK IF THE NEURONS THAT MODULATE AROUND AN EVENT ARE PLACE CELLS">
modulating_neuron_indices = {}
events = ['trial_inc', 'trial_dec', 'non_trial_inc', 'non_trial_dec', 'traj_inc', 'traj_dec']

for ev in events:
    if ev == 'trial_inc':
        modulating_neuron_indices[ev] = ti_increasing_neurons_on_trial_pokes.index.values
    if ev == 'trial_dec':
        modulating_neuron_indices[ev] = ti_decreasing_neurons_on_trial_pokes.index.values
    if ev == 'non_trial_inc':
        modulating_neuron_indices[ev] = ti_increasing_neurons_on_non_trial_pokes.index.values
    if ev == 'non_trial_dec':
        modulating_neuron_indices[ev] = ti_decreasing_neurons_on_non_trial_pokes.index.values
    if ev == 'traj_inc':
        modulating_neuron_indices[ev] = ti_increasing_neurons_on_start_ballistic.index.values
    if ev == 'traj_dec':
        modulating_neuron_indices[ev] = ti_decreasing_neurons_on_start_ballistic.index.values


venn3([set(modulating_neuron_indices['trial_dec']),
       set(modulating_neuron_indices['non_trial_dec']),
       set(modulating_neuron_indices['traj_dec'])],
      set_labels=('Successful trials poke', 'Non successful trials poke', 'Starts of ballistic trajectory'))
plt.title('Neurons that decrease activity')

venn3([set(modulating_neuron_indices['trial_inc']),
       set(modulating_neuron_indices['non_trial_inc']),
       set(modulating_neuron_indices['traj_inc'])],
      set_labels=('Successful trials poke', 'Non successful trials poke', 'Starts of ballistic trajectory'))
plt.title('Neurons that increase activity')


# </editor-fold>
# -------------------------------------------------
