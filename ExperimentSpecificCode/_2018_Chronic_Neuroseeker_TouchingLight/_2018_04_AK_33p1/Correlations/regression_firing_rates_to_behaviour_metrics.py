
from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import preprocessing

from BrainDataAnalysis import binning

import pandas as pd
import matplotlib.pyplot as plt

import sequence_viewer as sv
import drop_down as dd
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

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

shuffled_filenames = {'pb_dtp':'shuffled_mut_info_spike_rate_912_vs_distance_to_poke_patterned_behaviour.npy',
                      'pb_speed': 'shuffled_mut_info_spike_rate_522_vs_speed_patterned_behaviour.npy',
                      'dtp': 'shuffled_mut_info_spike_rate_33_vs_distance_to_poke.npy',
                      'speed': 'shuffled_mut_info_spike_rate_522_vs_speed.npy'}
# Load data
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)
spike_rates_0p25 = np.load(spike_rates_per_250ms_filename)

speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

distances_rat_to_poke_all_frames = np.load(join(patterned_vs_non_patterned_folder,
                                                'distances_rat_to_poke_all_frames.npy'))
speeds_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                               'speeds_patterned_behaviour_0p25.npy'))
distance_to_poke_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                                         'distance_to_poke_patterned_behaviour_0p25.npy'))
spike_rates_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                                    'spike_rates_patterned_behaviour_0p25.npy'))
speeds_non_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                                   'speeds_non_patterned_behaviour_0p25.npy'))
distance_to_poke_non_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                                             'distance_to_poke_non_patterned_behaviour_0p25.npy'))
spike_rates_non_patterned_behaviour_0p25 = np.load(join(patterned_vs_non_patterned_folder,
                                                        'spike_rates_non_patterned_behaviour_0p25.npy'))
windows_of_patterned_behaviour = np.load(join(patterned_vs_non_patterned_folder,
                                              'windows_of_patterned_behaviour.npy'), allow_pickle=True)
windows_of_non_patterned_behaviour = np.load(join(patterned_vs_non_patterned_folder,
                                                  'windows_of_non_patterned_behaviour.npy'), allow_pickle=True)

mi_pb_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'))
mi_non_pb_spikes_vs_speed = np.load(join(mutual_information_folder,
                                         'mutual_infos_spikes_vs_speed_non_patterned_behaviour.npy'))
mi_pb_spikes_vs_distance_to_poke = np.load(join(mutual_information_folder,
                                                'mutual_infos_spikes_vs_distance_to_poke_patterned_behaviour.npy'))
mi_non_pb_spikes_vs_distance_to_poke = np.load(
    join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke_non_patterned_behaviour.npy'))

mi_spikes_vs_distance_to_poke = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke.npy'))
mi_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected.npy'))


mis = {'pb_dtp': mi_pb_spikes_vs_distance_to_poke,
       'npb_dtp': mi_non_pb_spikes_vs_distance_to_poke,
       'pb_speed': mi_pb_spikes_vs_speed,
       'npb_speed': mi_non_pb_spikes_vs_speed,
       'speed': mi_spikes_vs_speed,
       'dtp': mi_spikes_vs_distance_to_poke}

mis_shuffled = {}
for s in shuffled_filenames:
    mis_shuffled[s] = np.load(join(mutual_information_folder, shuffled_filenames[s]))

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="GET THE NEURONS WITH SIGNIFICANT MUTUAL INFORMATION CONTENT">

confidence_level = 0.99
correlated_neuron_indices = {}
for s in mis_shuffled:
    mean_sh = np.mean(mis_shuffled[s])
    confi_intervals = mis_shuffled[s][int((1. - confidence_level) / 2 * 1000)], \
                      mis_shuffled[s][int((1. + confidence_level) / 2 * 1000)]

    correlated_neuron_indices[s] = np.squeeze(np.argwhere(mis[s] > mean_sh+confi_intervals[1]))
    if 'pb' in s:
        correlated_neuron_indices['n'+s] = np.squeeze(np.argwhere(mis['n'+s] > mean_sh + confi_intervals[1]))

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SIMPLE LINEAR REGRESSIONS">

# <editor-fold desc="SPEED REGRESSION">
X = spike_rates_0p25[correlated_neuron_indices['speed']].transpose()
Y = binning.rolling_window_with_step(speeds, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor_speed = linear_model.LinearRegression(normalize=True)
regressor_speed.fit(X_train, y_train)
print(regressor_speed.score(X_train, y_train))

Y_pred = regressor_speed.predict(X_test)
plt.plot(Y_pred)
plt.plot(y_test)
plt.scatter(Y_pred, y_test)
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')
# </editor-fold>

# <editor-fold desc="FULL DISTANCE TO POKE REGRESSION">
X = spike_rates_0p25[correlated_neuron_indices['dtp']].transpose()
Y = binning.rolling_window_with_step(distances_rat_to_poke_all_frames, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor_dtp = linear_model.LinearRegression(normalize=True)
regressor_dtp.fit(X_train, y_train)
print(regressor_dtp.score(X_train, y_train))

Y_pred = regressor_dtp.predict(X_test)
plt.plot(Y_pred)
plt.plot(y_test)
plt.scatter(Y_pred, y_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')
# </editor-fold>

# <editor-fold desc="PB VS NON PB DISTANCE TO POKE REGRESSION">
X_pb = spike_rates_patterned_behaviour_0p25[correlated_neuron_indices['pb_dtp']].transpose()
Y_pb = distance_to_poke_patterned_behaviour_0p25
X_pb_train, X_pb_test, y_pb_train, y_pb_test = train_test_split(X_pb, Y_pb, test_size=0.2, random_state=0)

regressor_pb_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_dtp.fit(X_pb_train, y_pb_train)
print(regressor_pb_dtp.score(X_pb_train, y_pb_train))

Y_pb_pred = regressor_pb_dtp.predict(X_pb_test)
plt.plot(Y_pb_pred)
plt.plot(y_pb_test)
plt.scatter(Y_pb_pred, y_pb_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')


X_npb = spike_rates_patterned_behaviour_0p25[correlated_neuron_indices['npb_dtp']].transpose()
Y_npb = distance_to_poke_patterned_behaviour_0p25
X_npb_train, X_npb_test, y_npb_train, y_npb_test = train_test_split(X_npb, Y_npb, test_size=0.2, random_state=0)

regressor_npb_dtp = linear_model.LinearRegression(normalize=True)
regressor_npb_dtp.fit(X_npb_train, y_npb_train)
print(regressor_npb_dtp.score(X_npb_train, y_npb_train))

Y_npb_pred = regressor_npb_dtp.predict(X_npb_test)
plt.plot(Y_npb_pred)
plt.plot(y_npb_test)
plt.scatter(Y_npb_pred, y_npb_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')
# </editor-fold>

# </editor-fold>
# -------------------------------------------------


common_pb_npb_dtp_neuron_indices = np.intersect1d(correlated_neuron_indices['pb_dtp'], correlated_neuron_indices['npb_dtp'])

correlated_neuron_indices_unique_pb_dtp = np.delete(correlated_neuron_indices['pb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['pb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))
correlated_neuron_indices_unique_npb_dtp = np.delete(correlated_neuron_indices['npb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['npb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))

X_pb = spike_rates_patterned_behaviour_0p25[correlated_neuron_indices_unique_npb_dtp].transpose()
Y_pb = distance_to_poke_patterned_behaviour_0p25
X_pb_train, X_pb_test, y_pb_train, y_pb_test = train_test_split(X_pb, Y_pb, test_size=0.2, random_state=0)

regressor_pb_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_dtp.fit(X_pb_train, y_pb_train)
print(regressor_pb_dtp.score(X_pb_train, y_pb_train))