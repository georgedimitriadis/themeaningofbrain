
from os.path import join
import numpy as np
import pickle
import bisect

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, regression_functions as reg_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp

from sklearn import linear_model, metrics, model_selection, preprocessing

from npeet.lnc import MI

import pygam as pg

import sequence_viewer as sv
import transform as tr
import drop_down as dd

from BrainDataAnalysis import binning

import pandas as pd
import matplotlib.pyplot as plt

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

regressions_folder = join(results_folder, 'Regressions', 'DistanceTraveledBetweenPokes')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

# Load data
spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)

distances_rat_head_to_poke_smoothed = np.load(join(regressions_folder, 'distances_rat_head_to_poke_smoothed.npy'))

frame_regions_away_from_poke = np.load(join(regressions_folder, 'frame_regions_away_from_poke.npy'))

distance_traveled_from_poke = np.load(join(regressions_folder, 'distance_traveled_from_poke.npy'))

distance_traveled_from_poke_in_periods = np.load(join(regressions_folder, 'distance_traveled_from_poke_in_periods.npy'),
                                                allow_pickle=True)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="COMMON VARIABLES">
poke_position = [0.9, 0]

#   Smoothing to do the regressions
fr_smooth_time = 0.15
fr_smooth_frames = fr_smooth_time * 120

poking_threshold = 0.149


# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="PREPARE THE DATA FOR THE MUTUAL INFORMATION CALC (AND THE FRAME WIDNOWS THAT THE PERIODS ARE IN)">

frame_windows_away_from_poke = np.empty(0)
frame_windows_away_from_poke_in_periods = []
duration_in_frames_of_travel_periods = []
for f in frame_regions_away_from_poke:
    period_frames = np.arange(f[0], f[1]).astype(np.int)
    frame_windows_away_from_poke_in_periods.append(period_frames)
    duration_in_frames_of_travel_periods.append(len(period_frames))
    frame_windows_away_from_poke = np.concatenate((frame_windows_away_from_poke, period_frames))
frame_windows_away_from_poke = frame_windows_away_from_poke.astype(np.int)


spike_rates_away_from_poke = spike_rates[:, frame_windows_away_from_poke]
spike_rates_away_from_poke_smoothed = binning.rolling_window_with_step(spike_rates_away_from_poke,
                                                                       np.mean, fr_smooth_frames, fr_smooth_frames)

distance_traveled_from_poke_smoothed = binning.rolling_window_with_step(distance_traveled_from_poke,
                                                              np.mean, fr_smooth_frames, fr_smooth_frames)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="DO THE MIs (RUN ONCE)">
# NOTE The MIs were done with fr_smooth_time = 0.5
n = 0
mi_spike_rates_vs_distance_traveled_from_poke = []
for rate in spike_rates_away_from_poke_smoothed:
    mi_spike_rates_vs_distance_traveled_from_poke.append(MI.mi_LNC([rate.tolist(), distance_traveled_from_poke_smoothed],
                                                                  k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spike_rates_vs_distance_traveled_from_poke = np.array(mi_spike_rates_vs_distance_traveled_from_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_traveled_from_poke.npy'),
        mi_spike_rates_vs_distance_traveled_from_poke)


#   Do the shuffle of the best fr vs distance to travel to next poke
max_neuron = np.argmax(mi_spike_rates_vs_distance_traveled_from_poke)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_away_from_poke_smoothed[max_neuron],
                                                 distance_traveled_from_poke_smoothed,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder,
             'shuffled_mut_info_spike_rate_{}_vs_distance_traveled_from_poke.npy'.format(str(max_neuron))),
        shuffled)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS OF THE MI CALCULATIONS">

#   Load the MIs and the shuffle
distance_traveled_from_poke_smoothed = np.array(distance_traveled_from_poke_smoothed)

mi_spike_rates_vs_distance_traveled_from_poke = \
    np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_traveled_from_poke.npy'))
max_neuron = np.argmax(mi_spike_rates_vs_distance_traveled_from_poke)
shuffled = np.load(join(mutual_information_folder,
                        'shuffled_mut_info_spike_rate_{}_vs_distance_traveled_from_poke.npy'.
                        format(str(max_neuron))))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]

#   Have a look at the MIs vs the chance hist
#       Log
_ = plt.hist(mi_spike_rates_vs_distance_traveled_from_poke, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
_ = plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(0, 0, 1, 1))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
#       Linear
_ = plt.hist(mi_spike_rates_vs_distance_traveled_from_poke, bins= 200, color=(0, 0, 1, 1))
_ = plt.hist(shuffled, bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, confi_intervals[0], confi_intervals[1]], 0, 20)

#   Where are the neurons
high_corr_neurons_index = np.squeeze(np.argwhere(mi_spike_rates_vs_distance_traveled_from_poke > confi_intervals[1]))
high_corr_neurons = template_info.loc[high_corr_neurons_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=high_corr_neurons,
                                     dot_sizes=mi_spike_rates_vs_distance_traveled_from_poke[high_corr_neurons_index] * 4000,
                                     font_size=5)

#   Look at individual spike rates vs the distance traveled
hipp_neuron_index = template_info[template_info['template number'] == 89].index.values[0]
neuron_index_to_show = hipp_neuron_index
neuron_index_to_show = max_neuron
plt.plot((distance_traveled_from_poke_smoothed - distance_traveled_from_poke_smoothed.min()) /
         (distance_traveled_from_poke_smoothed.max() - distance_traveled_from_poke_smoothed.min()))
plt.plot((spike_rates_away_from_poke_smoothed[neuron_index_to_show] - spike_rates_away_from_poke_smoothed[neuron_index_to_show].min()) /
         (spike_rates_away_from_poke_smoothed[neuron_index_to_show].max() - spike_rates_away_from_poke_smoothed[neuron_index_to_show].min()))

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SET UP THE REGRESSION DATA FOR THE JUST LEAVING THE POKE PART (FIRST PART OF THE PERIOD)">

#   Use only a few seconds as the rat approaches the poke
distance_traveled_from_poke_smoothed = np.array(distance_traveled_from_poke_smoothed)

period_duration_to_select_over = 5  # Use this amount of seconds before the poke
min_period_duration_to_keep = 3  # Ignore periods shorter than this
period_duration_to_select_over_in_frames = period_duration_to_select_over * 120
min_period_duration_to_keep_in_frames = min_period_duration_to_keep * 120

frame_windows_away_from_poke_leaving_poke = np.empty(0)
distance_traveled_from_poke_close_to_poke = np.empty(0)
starts_of_periods = [0]
lengths_of_periods = []
for w in np.arange(len(frame_windows_away_from_poke_in_periods)):
    frames = np.array(frame_windows_away_from_poke_in_periods[w])
    distances = np.array(distance_traveled_from_poke_in_periods[w])
    if len(frames) > min_period_duration_to_keep_in_frames:
        frame_windows_away_from_poke_leaving_poke = \
            np.concatenate((frame_windows_away_from_poke_leaving_poke, frames[:period_duration_to_select_over_in_frames]))
        distance_traveled_from_poke_close_to_poke = \
            np.concatenate((distance_traveled_from_poke_close_to_poke, distances[:period_duration_to_select_over_in_frames]))
        starts_of_periods.append(len(frame_windows_away_from_poke_leaving_poke) + 1)
        lengths_of_periods.append(len(frames[-period_duration_to_select_over_in_frames:]))

frame_windows_away_from_poke_leaving_poke = frame_windows_away_from_poke_leaving_poke.astype(np.int)
spike_rates_away_from_poke_close_to_poke = spike_rates[:, frame_windows_away_from_poke_leaving_poke]
starts_of_periods = np.delete(starts_of_periods, -1)

distance_traveled_from_poke_close_to_poke_smoothed = \
    np.array(binning.rolling_window_with_step(distance_traveled_from_poke_close_to_poke,
                                              np.mean, fr_smooth_frames, fr_smooth_frames))

spike_rates_away_from_poke_close_to_poke_smoothed = \
    np.array(binning.rolling_window_with_step(spike_rates_away_from_poke_close_to_poke,
                                              np.mean, fr_smooth_frames, fr_smooth_frames))

'''

plt.plot((distance_to_travel_to_poke_close_to_poke_smoothed - distance_to_travel_to_poke_close_to_poke_smoothed.min()) /
         (distance_to_travel_to_poke_close_to_poke_smoothed.max() - distance_to_travel_to_poke_close_to_poke_smoothed.min()))
plt.plot((spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show] -
          spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].min()) /
         (spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].max() -
          spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].min()))

'''
# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SET UP THE REGRESSION DATA FOR THE ARRIVING AT THE POKE PART (FINAL PART OF THE PERIOD)">

#   Use only a few seconds as the rat approaches the poke
distance_traveled_from_poke_smoothed = np.array(distance_traveled_from_poke_smoothed)

period_duration_to_select_over = 10  # Use this amount of seconds before the poke
min_period_duration_to_keep = 5  # Ignore periods shorter than this
period_duration_to_select_over_in_frames = period_duration_to_select_over * 120
min_period_duration_to_keep_in_frames = min_period_duration_to_keep * 120

frame_windows_away_from_poke_arriving_to_poke = np.empty(0)
distance_to_travel_to_poke_close_to_poke = np.empty(0)
starts_of_periods = [0]
lengths_of_periods = []
for w in np.arange(len(frame_windows_away_from_poke_in_periods)):
    frames = np.array(frame_windows_away_from_poke_in_periods[w])
    distances = np.array(distance_traveled_from_poke_in_periods[w])
    if len(frames) > min_period_duration_to_keep_in_frames:
        frame_windows_away_from_poke_arriving_to_poke = \
            np.concatenate((frame_windows_away_from_poke_arriving_to_poke, frames[-period_duration_to_select_over_in_frames:]))
        distance_to_travel_to_poke_close_to_poke = \
            np.concatenate((distance_to_travel_to_poke_close_to_poke, distances[-period_duration_to_select_over_in_frames:]))
        starts_of_periods.append(len(frame_windows_away_from_poke_arriving_to_poke) + 1)
        lengths_of_periods.append(len(frames[-period_duration_to_select_over_in_frames:]))

frame_windows_away_from_poke_arriving_to_poke = frame_windows_away_from_poke_arriving_to_poke.astype(np.int)
spike_rates_away_from_poke_close_to_poke = spike_rates[:, frame_windows_away_from_poke_arriving_to_poke]
starts_of_periods = np.delete(starts_of_periods, -1)

distance_to_travel_to_poke_close_to_poke_smoothed = \
    np.array(binning.rolling_window_with_step(distance_to_travel_to_poke_close_to_poke,
                                              np.mean, fr_smooth_frames, fr_smooth_frames))

spike_rates_away_from_poke_close_to_poke_smoothed = \
    np.array(binning.rolling_window_with_step(spike_rates_away_from_poke_close_to_poke,
                                              np.mean, fr_smooth_frames, fr_smooth_frames))

'''

plt.plot((distance_to_travel_to_poke_close_to_poke_smoothed - distance_to_travel_to_poke_close_to_poke_smoothed.min()) /
         (distance_to_travel_to_poke_close_to_poke_smoothed.max() - distance_to_travel_to_poke_close_to_poke_smoothed.min()))
plt.plot((spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show] -
          spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].min()) /
         (spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].max() -
          spike_rates_away_from_poke_close_to_poke_smoothed[neuron_index_to_show].min()))

'''
# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="RUN THE REGRESSION">
#   Get the relevant neurons
mi_spike_rates_vs_distance_traveled_from_poke = \
    np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_traveled_from_poke.npy'))
max_neuron = np.argmax(mi_spike_rates_vs_distance_traveled_from_poke)
shuffled = np.load(join(mutual_information_folder,
                        'shuffled_mut_info_spike_rate_{}_vs_distance_traveled_from_poke.npy'.
                        format(str(max_neuron))))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]

high_corr_neurons_index = np.squeeze(np.argwhere(mi_spike_rates_vs_distance_traveled_from_poke > 0.005 + confi_intervals[1]))

#   Standartize X, Y
Y = preprocessing.StandardScaler().\
    fit_transform(np.reshape(distance_to_travel_to_poke_close_to_poke_smoothed, (-1, 1))).reshape(-1)

X = preprocessing.StandardScaler().\
    fit_transform(spike_rates_away_from_poke_close_to_poke_smoothed[high_corr_neurons_index].transpose())

#   Set up the regressors
model_gam = pg.LinearGAM()

#       Split randomly by periods
train_periods_indices = np.sort(np.random.choice(np.arange(len(starts_of_periods)), int(len(starts_of_periods) * 0.8),
                                                 replace=False).astype(np.int))
test_period_indices = np.delete(np.arange(len(starts_of_periods)), train_periods_indices)
starts_of_periods_smoothed = (starts_of_periods / fr_smooth_frames).astype(np.int)
lengths_of_periods_smoothed = (np.array(lengths_of_periods) / fr_smooth_frames).astype(np.int)

train_periods = np.empty(0).astype(np.int)
test_periods = np.empty(0).astype(np.int)
for i in train_periods_indices:
    train_periods = np.concatenate((train_periods,
                                    np.arange(starts_of_periods_smoothed[i],
                                              starts_of_periods_smoothed[i] + lengths_of_periods_smoothed[i])))
for i in test_period_indices:
    test_periods = np.concatenate((test_periods,
                                   np.arange(starts_of_periods_smoothed[i],
                                             starts_of_periods_smoothed[i] + lengths_of_periods_smoothed[i])))

X_used = X #  X or X_poly (for poly use alpha = 0.062)
X_train = X_used[train_periods, :]
Y_train = Y[train_periods]
X_test = X_used[test_periods, :]
Y_test = Y[test_periods]


#   Run the GAM regression
# Fit

model = model_gam
model.gridsearch(X_train, Y_train)

pickle.dump(model, open(join(regressions_folder, 'linear_GAM_frs_099_vs_distance_traveled_from_poke.pcl'), "wb"))
model = pickle.load(open(join(regressions_folder, 'linear_GAM_frs_095_vs_distance_to_travel_to_poke.pcl'), "rb"))

# Show partial dependence of each neuron
signif_thres = 1e-2
neurons_p_values = np.array(model.statistics_['p_values'])
sig_neurons_index = np.squeeze(np.argwhere(neurons_p_values < signif_thres))
num_sig_neurons = len(sig_neurons_index)

max_cols = 4
rows = np.ceil(num_sig_neurons/max_cols)
cols = num_sig_neurons if num_sig_neurons < max_cols + 1 else max_cols
fig = plt.figure(0)
for i in np.arange(len(sig_neurons_index)):
    ax = fig.add_subplot(rows, cols, i + 1)
    t = int(sig_neurons_index[i])
    x_axis = model.generate_X_grid(term=t)
    ax.plot(x_axis[:, t], model.partial_dependence(t, x_axis))
    ax.plot(x_axis[:, t], model.partial_dependence(t, x_axis, width=0.99)[1], c='r', ls='--')


plt.plot(Y_test)
plt.plot(model.predict(X_test))

plt.plot(Y_train)
plt.plot(model.predict(X_train))

# </editor-fold>
# -------------------------------------------------