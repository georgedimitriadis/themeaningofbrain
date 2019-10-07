
from os.path import join
import numpy as np
import pickle

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import pipeline

from BrainDataAnalysis.Statistics import binning

import pandas as pd
import matplotlib.pyplot as plt

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

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')

regressions_folder = join(results_folder, 'Regressions')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

shuffled_filenames = {'pb_dtp':'shuffled_mut_info_spike_rate_35_vs_distance_to_poke_patterned_behaviour.npy',
                      'pb_speed': 'shuffled_mut_info_spike_rate_522_vs_speed_patterned_behaviour.npy',
                      'dtp': 'shuffled_mut_info_spike_rate_33_vs_distance_to_poke.npy',
                      'speed': 'shuffled_mut_info_spike_rate_522_vs_speed.npy'}
# Load data
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cortex_sorting.df'))

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

# <editor-fold desc="SPEED REGRESSION V0">
X = spike_rates_0p25[correlated_neuron_indices['speed']].transpose()
X = spike_rates_0p25[522].transpose()
Y = binning.rolling_window_with_step(speeds, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor_speed = linear_model.LinearRegression(normalize=True)
regressor_speed.fit(np.transpose([X_train]), y_train)
print(regressor_speed.score(np.transpose([X_train]), y_train))

Y_pred = regressor_speed.predict(np.transpose([X_test]))
plt.plot(Y_pred)
plt.plot(y_test)
plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')
# </editor-fold>

# <editor-fold desc="SPEED REGRESSION V1">
X = spike_rates_0p25[522].transpose()
Y = binning.rolling_window_with_step(speeds, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor_speed = linear_model.LinearRegression(normalize=True)
regressor_speed.fit(np.transpose([X_train]), y_train)
print(regressor_speed.score(np.transpose([X_train]), y_train))

Y_pred = regressor_speed.predict(np.transpose([X_test]))
plt.plot(Y_pred)
plt.plot(y_test)
plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')
# </editor-fold>

# <editor-fold desc="SPEED REGRESSION V2">
from sklearn.preprocessing import StandardScaler
X = spike_rates_0p25[correlated_neuron_indices['speed']].transpose()
scaler = StandardScaler()
X = scaler.fit_transform(X)
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


# <editor-fold desc="SPEED REGRESSION V3">
X = spike_rates_0p25[correlated_neuron_indices['speed']].transpose()
Y = binning.rolling_window_with_step(speeds, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor_speed = linear_model.LinearRegression(normalize=True)
regressor_speed.fit(X_train, y_train)
print(regressor_speed.score(X_train, y_train))

Y_pred = regressor_speed.predict(X_test)
plt.plot(Y_pred)
plt.plot(np.array(y_test))
plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([-25, -10, 0, 10, 25], [-25, -10, 0, 10, 25], c='k')

plt.scatter(Y_pred, 1.2 * np.array(y_test))
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')
# </editor-fold>

# <editor-fold desc="SPEED REGRESSION V4 POLYNOMIAL">
X = spike_rates_0p25[correlated_neuron_indices['speed']].transpose()
Y = binning.rolling_window_with_step(speeds, np.nanmean, 30, 30)[:-1]

X_train = X[:10000, :]
y_train = Y[:10000]
X_test = X[:-10000, :]
y_test = Y[:-10000]

model = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

Y_pred = model.predict(X_test)
plt.plot(Y_pred)
plt.plot(np.array(y_test))
plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([-25, -10, 0, 10, 25], [-25, -10, 0, 10, 25], c='k')

plt.scatter(Y_pred, y_test)
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')
# </editor-fold>

# <editor-fold desc="FULL DISTANCE TO POKE REGRESSION">
X = spike_rates_0p25[correlated_neuron_indices['dtp']].transpose()
Y = binning.rolling_window_with_step(distances_rat_to_poke_all_frames, np.nanmean, 30, 30)[:-1]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train = X[:10000, :]
y_train = Y[:10000]
X_test = X[10000:, :]
y_test = Y[10000:]

#   Linear
regressor_dtp = linear_model.LinearRegression(normalize=True)
regressor_dtp.fit(X_train, y_train)
print(regressor_dtp.score(X_train, y_train))
print(regressor_dtp.score(X_test, y_test))

Y_pred = regressor_dtp.predict(X_test)

#   Polynomial
model_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
file_model_dtp = join(regressions_folder, 'regression_dtp_Linear_2nd_Order.pcl')


model_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=None, fit_intercept=True))
file_model_dtp = join(regressions_folder, 'regression_dtp_Lasso_2nd_Order.pcl')

model_dtp.fit(X_train, y_train)

pickle.dump(model_dtp, open(file_model_dtp, "wb"))

print(model_dtp.score(X_train, y_train))
print(model_dtp.score(X_test, y_test))

Y_pred = model_dtp.predict(X_test)


plt.plot(Y_pred)
plt.plot(y_test)

plt.plot(y_train)
plt.plot(model_dtp.predict(X_train))

plt.scatter(Y_pred, y_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')

plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([-2, 0, 2], [-2, 0, 2], c='k')


#   Regression using random neurons that do not have any mutual info with the dtp
non_correlated_neurons_to_dtp = np.random.choice([d for d in np.arange(correlated_neuron_indices['dtp'].max())
                                                  if d not in correlated_neuron_indices['dtp']],
                                                 len(correlated_neuron_indices['dtp']), replace=False)
X = spike_rates_0p25[non_correlated_neurons_to_dtp].transpose()
Y = binning.rolling_window_with_step(distances_rat_to_poke_all_frames, np.nanmean, 30, 30)[:-1]

X_train = X[:10000, :]
y_train = Y[:10000]
X_test = X[:-10000, :]
y_test = Y[:-10000]

model_random_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_random_dtp = pipeline.make_pipeline(PolynomialFeatures(2),
                                   linear_model.LassoCV(cv=None, fit_intercept=True))
model_random_dtp.fit(X_train, y_train)

file_model_random_dtp = join(regressions_folder, 'regression_random_dtp_Lasso_2nd_Order.pcl')
pickle.dump(model_random_dtp, open(file_model_random_dtp, "wb"))

model_random_dtp = pickle.load(open(file_model_random_dtp, "rb"))
print(model_random_dtp.score(X_train, y_train))
print(model_random_dtp.score(X_test, y_test))

Y_pred = model_random_dtp.predict(X_test)
plt.plot(y_test)
plt.plot(Y_pred)


# </editor-fold>

# <editor-fold desc="PB VS NON PB DISTANCE TO POKE REGRESSION">
X_pb_dtp = spike_rates_patterned_behaviour_0p25[correlated_neuron_indices['pb_dtp']].transpose()
Y_pb_dtp = distance_to_poke_patterned_behaviour_0p25
# X_pb_train, X_pb_test, y_pb_train, y_pb_test = train_test_split(X_pb_dtp, Y_pb_dtp, test_size=0.2, random_state=0)

X_pb_dtp_train = X_pb_dtp[:200, :]
y_pb_dtp_train = Y_pb_dtp[:200]
X_pb_dtp_test = X_pb_dtp[200:, :]
y_pb_dtp_test = Y_pb_dtp[200:]

#   Linear Patterned Behaviour
regressor_pb_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_dtp.fit(X_pb_dtp_train, y_pb_dtp_train)
print(regressor_pb_dtp.score(X_pb_dtp_train, y_pb_dtp_train))
print(regressor_pb_dtp.score(X_pb_dtp_test, y_pb_dtp_test))

Y_pb_dtp_pred = regressor_pb_dtp.predict(X_pb_dtp_train)

#   Polynomial Patterned Behaviour
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=None, fit_intercept=True))

model_pb_dtp.fit(X_pb_dtp_train, y_pb_dtp_train)

print(model_pb_dtp.score(X_pb_dtp_train, y_pb_dtp_train))
print(model_pb_dtp.score(X_pb_dtp_test, y_pb_dtp_test))

Y_pb_dtp_pred = model_pb_dtp.predict(X_pb_dtp_test)

plt.plot(Y_pb_dtp_pred)
plt.plot(y_pb_dtp_test)

plt.plot(y_pb_dtp_train)
plt.plot(model_pb_dtp.predict(X_pb_dtp_train))

plt.scatter(Y_pb_dtp_pred, y_pb_dtp_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')

# </editor-fold>

# </editor-fold>
# -------------------------------------------------
