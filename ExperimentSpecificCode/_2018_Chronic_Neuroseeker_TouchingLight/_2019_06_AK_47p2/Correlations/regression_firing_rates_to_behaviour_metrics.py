
from os.path import join
import numpy as np
import pickle

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import pipeline
from sklearn import svm

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
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')

regressions_folder = join(results_folder, 'Regressions')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

shuffled_filenames = {'pb_dtp':'shuffled_mut_info_spike_rate_912_vs_distance_to_poke_patterned_behaviour.npy',
                      'pb_speed': 'shuffled_mut_info_spike_rate_589_vs_speed_patterned_behaviour.npy',
                      'dtp': 'shuffled_mut_info_spike_rate_960_vs_distance_to_poke.npy',
                      'speed': 'shuffled_mut_info_spike_rate_1140_vs_speed.npy'}
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
                                              'windows_of_patterned_behaviour_list.npy'), allow_pickle=True)
windows_of_non_patterned_behaviour = np.load(join(patterned_vs_non_patterned_folder,
                                                  'windows_of_non_patterned_behaviour_list.npy'), allow_pickle=True)

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
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


X_train = X[:10000, :]
y_train = Y[:10000]
X_test = X[10000:, :]
y_test = Y[10000:]

#   Linear
regressor_speed = linear_model.LinearRegression(normalize=True)
regressor_speed.fit(X_train, y_train)
print(regressor_speed.score(X_train, y_train))
print(regressor_speed.score(X_test, y_test))

Y_pred = regressor_speed.predict(X_test)

#   Polynomial
model_speed = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_speed.fit(X_train, y_train)

print(model_speed.score(X_train, y_train))
print(model_speed.score(X_test, y_test))

Y_pred = model_speed.predict(X_test)

plt.plot(Y_pred)
plt.plot(y_test)
plt.scatter(Y_pred, y_test)
plt.plot([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], c='k')

plt.scatter(np.diff(Y_pred), np.diff(y_test))
plt.plot([-25, -10, 0, 10, 25], [-25, -10, 0, 10, 25], c='k')
# </editor-fold>

# <editor-fold desc="FULL DISTANCE TO POKE REGRESSION">
X = spike_rates_0p25[correlated_neuron_indices['dtp']].transpose()
X = spike_rates_0p25[correlated_neuron_indices['pb_dtp']].transpose()
Y = binning.rolling_window_with_step(distances_rat_to_poke_all_frames, np.nanmean, 30, 30)[:-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

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

model_dtp = pipeline.make_pipeline(PolynomialFeatures(2),
                                   linear_model.RidgeCV(alphas=np.array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01,
                                                                         1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]),
                                                        cv=None, fit_intercept=True, gcv_mode=None, normalize=False,
                                                        scoring=None, store_cv_values=False))

model_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=5, fit_intercept=True))
file_model_dtp = join(regressions_folder, 'regression_dtp_Lasso_2nd_Order.pcl')


model_dtp.fit(X_train, y_train)


pickle.dump(model_dtp, open(file_model_dtp, "wb"))

print(model_dtp.score(X_train, y_train))
print(model_dtp.score(X_test, y_test))

Y_pred = model_dtp.predict(X_test)

plt.plot(y_test)
plt.plot(Y_pred)
plt.plot(binning.rolling_window_with_step(Y_pred, np.mean, 20, 1))

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
X_test = X[10000:, :]
y_test = Y[10000:]

model_random_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
file_model_random_dtp = join(regressions_folder, 'regression_random_dtp_Linear_2nd_Order.pcl')

model_random_dtp = pipeline.make_pipeline(PolynomialFeatures(2),
                                   linear_model.RidgeCV(alphas=np.array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01,
                                                                         1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]),
                                                        cv=None, fit_intercept=True, gcv_mode=None, normalize=False,
                                                        scoring=None, store_cv_values=False))
model_random_dtp = pipeline.make_pipeline(PolynomialFeatures(2),
                                   linear_model.LassoCV(alphas=np.array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01,
                                                                         1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]),
                                                        cv=None, fit_intercept=True))
file_model_random_dtp = join(regressions_folder, 'regression_random_dtp_Lasso_2nd_Order.pcl')

model_random_dtp.fit(X_train, y_train)


pickle.dump(model_random_dtp, open(file_model_random_dtp, "wb"))

model_random_dtp = pickle.load(open(file_model_random_dtp, "rb"))
print(model_random_dtp.score(X_train, y_train))
print(model_random_dtp.score(X_test, y_test))

Y_pred = model_random_dtp.predict(X_test)
plt.plot(y_test)
plt.plot(Y_pred)
plt.plot(binning.rolling_window_with_step(Y_pred, np.mean, 10, 1))

plt.plot(y_train)
plt.plot(model_dtp.predict(X_train))

# </editor-fold>

# <editor-fold desc="PB VS NON PB DISTANCE TO POKE REGRESSION">
smoothing_time = 0.2
smoothing_frames = int(smoothing_time / 0.00833)
spike_rates_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates[:, windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)
distance_to_poke_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)

spike_rates_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates_patterned_behaviour_smoothed, np.mean, 5, 1)
distance_to_poke_patterned_behaviour_smoothed = \
    np.array(distance_to_poke_patterned_behaviour_smoothed[len(distance_to_poke_patterned_behaviour_smoothed)
             - spike_rates_patterned_behaviour_smoothed.shape[1]:])

#   Patterned Behaviour
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices['pb_dtp']].transpose()
Y_pb_dtp = distance_to_poke_patterned_behaviour_smoothed

# X_pb_dtp_train, X_pb_dtp_test, y_pb_dtp_train, y_pb_dtp_test = train_test_split(X_pb_dtp, Y_pb_dtp, test_size=0.3, random_state=0)

train_size = int(0.7 * spike_rates_patterned_behaviour_smoothed.shape[1])
X_pb_dtp_train = X_pb_dtp[:train_size, :]
y_pb_dtp_train = Y_pb_dtp[:train_size]
X_pb_dtp_test = X_pb_dtp[train_size:, :]
y_pb_dtp_test = Y_pb_dtp[train_size:]

s = len(distance_to_poke_patterned_behaviour_smoothed)
test_size = int(0.1 * s / 2)
starts = [int(np.random.choice(np.arange(s/2 - test_size))), int(np.random.choice(np.arange(s/2, s - test_size)))]
test_samples = np.concatenate((np.arange(starts[0], starts[0] + test_size), np.arange(starts[1], starts[1] + test_size)))
train_samples = np.delete(np.arange(s), test_samples)

X_pb_dtp_train = X_pb_dtp[train_samples, :]
y_pb_dtp_train = np.array(Y_pb_dtp)[train_samples]
X_pb_dtp_test = X_pb_dtp[test_samples, :]
y_pb_dtp_test = np.array(Y_pb_dtp)[test_samples]


#   Linear Patterned Behaviour
regressor_pb_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_dtp.fit(X_pb_dtp_train, y_pb_dtp_train)
print(regressor_pb_dtp.score(X_pb_dtp_train, y_pb_dtp_train))
print(regressor_pb_dtp.score(X_pb_dtp_test, y_pb_dtp_test))

Y_pb_dtp_pred = regressor_pb_dtp.predict(X_pb_dtp_test)

#   Polynomial Patterned Behaviour
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=None, fit_intercept=True))
model_pb_dtp = svm.SVR(kernel='poly', degree=2, C=1, gamma=0.1, epsilon=.001)

model_pb_dtp.fit(X_pb_dtp_train, y_pb_dtp_train)

print(model_pb_dtp.score(X_pb_dtp_train, y_pb_dtp_train))
print(model_pb_dtp.score(X_pb_dtp_test, y_pb_dtp_test))

Y_pb_dtp_pred = model_pb_dtp.predict(X_pb_dtp_test)

plt.plot(Y_pb_dtp_pred)
plt.plot(y_pb_dtp_test)
plt.plot(binning.rolling_window_with_step(Y_pb_dtp_pred, np.mean, 10, 1))

plt.plot(y_pb_dtp_train)
plt.plot(model_pb_dtp.predict(X_pb_dtp_train))

plt.scatter(Y_pb_dtp_pred, y_pb_dtp_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')

#   Do Patterned Behaviour in a loop (leave a few out) to generate the prediction for the whole series

X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices['pb_dtp']].transpose()
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices['dtp']].transpose()

Y_pb_dtp = distance_to_poke_patterned_behaviour_smoothed
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())

leave_percentage_out = 0.001
s = len(distance_to_poke_patterned_behaviour_smoothed)
leave_amount_out = int(leave_percentage_out * s)

Y_pb_dtp_pred = np.empty(0)
start_sample = 0
while start_sample + leave_amount_out <= s:
    test_samples = np.arange(start_sample, start_sample + leave_amount_out)
    train_samples = np.delete(np.arange(s), test_samples)
    start_sample += leave_amount_out
    print(start_sample, start_sample+leave_amount_out)

    X_pb_dtp_train = X_pb_dtp[train_samples, :]
    y_pb_dtp_train = np.array(Y_pb_dtp)[train_samples]
    X_pb_dtp_test = X_pb_dtp[test_samples, :]
    y_pb_dtp_test = np.array(Y_pb_dtp)[test_samples]

    model_pb_dtp.fit(X_pb_dtp_train, y_pb_dtp_train)
    Y_pb_dtp_pred = np.concatenate((Y_pb_dtp_pred, model_pb_dtp.predict(X_pb_dtp_test)))

error = np.linalg.norm(Y_pb_dtp[:len(Y_pb_dtp_pred)] - Y_pb_dtp_pred)
print(error)

dtp_true_line = plt.plot(Y_pb_dtp)
pb_dtp_pred_line = plt.plot(Y_pb_dtp_pred)

#   Non Patterned Behaviour
X_npb = spike_rates_non_patterned_behaviour_0p25[correlated_neuron_indices['npb_dtp']].transpose()
X_npb = spike_rates_non_patterned_behaviour_0p25[correlated_neuron_indices['dtp']].transpose()

Y_npb = distance_to_poke_non_patterned_behaviour_0p25

X_npb_train = X_npb[:700, :]
y_npb_train = Y_npb[:700]
X_npb_test = X_npb[700:, :]
y_npb_test = Y_npb[700:]

#   Linear Non Patterned Behaviour
regressor_npb_dtp = linear_model.LinearRegression(normalize=True)
regressor_npb_dtp.fit(X_npb_train, y_npb_train)
print(regressor_npb_dtp.score(X_npb_train, y_npb_train))
print(regressor_npb_dtp.score(X_npb_test, y_npb_test))

Y_npb_pred = regressor_npb_dtp.predict(X_npb_test)

#   Polynomial Non Patterned Behaviour
model_npb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_npb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=None, fit_intercept=True))
model_npb_dtp = svm.SVR(kernel='rbf', degree=2, C=0.001, gamma=0.1, epsilon=.1)

model_npb_dtp.fit(X_npb_train, y_npb_train)

print(model_npb_dtp.score(X_npb_train, y_npb_train))
print(model_npb_dtp.score(X_npb_test, y_npb_test))

Y_npb_pred = model_npb_dtp.predict(X_npb_test)


plt.plot(Y_npb_pred)
plt.plot(y_npb_test)

plt.plot(y_npb_train)
plt.plot(model_npb_dtp.predict(X_npb_train))

plt.scatter(Y_npb_pred, y_npb_test)
plt.plot([0, 1, 2], [0, 1, 2], c='k')

#   Unique patterned and non patterned beahviour

common_pb_npb_dtp_neuron_indices = np.intersect1d(correlated_neuron_indices['pb_dtp'], correlated_neuron_indices['npb_dtp'])

correlated_neuron_indices_unique_pb_dtp = np.delete(correlated_neuron_indices['pb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['pb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))
correlated_neuron_indices_unique_npb_dtp = np.delete(correlated_neuron_indices['npb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['npb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))

#   Patterned unique
X_pb_unique_dtp = spike_rates_patterned_behaviour_0p25[common_pb_npb_dtp_neuron_indices].transpose()
Y_pb_unique_dtp = distance_to_poke_patterned_behaviour_0p25


X_pb_unique_train = X_pb_unique_dtp[:700, :]
y_pb_unique_train = Y_pb_unique_dtp[:700]
X_pb_unique_test = X_pb_unique_dtp[700:, :]
y_pb_unique_test = Y_pb_unique_dtp[700:]

#    Linear
regressor_pb_uniqu_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_uniqu_dtp.fit(X_pb_unique_train, y_pb_unique_train)
print(regressor_pb_uniqu_dtp.score(X_pb_unique_train, y_pb_unique_train))
print(regressor_pb_uniqu_dtp.score(X_pb_unique_test, y_pb_unique_test))


Y_pb_uniqu_dtp_pred = regressor_npb_dtp.predict(X_pb_unique_test)

#   Polynomial
model_pb_uniqu_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_pb_uniqu_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=3, fit_intercept=True))

model_pb_uniqu_dtp.fit(X_pb_unique_train, y_pb_unique_train)
print(model_pb_uniqu_dtp.score(X_pb_unique_train, y_pb_unique_train))
print(model_pb_uniqu_dtp.score(X_pb_unique_test, y_pb_unique_test))

Y_pb_uniqu_dtp_pred = model_pb_uniqu_dtp.predict(X_pb_unique_test)


plt.plot(Y_pb_uniqu_dtp_pred)
plt.plot(y_pb_unique_test)


#   Non patterned unique
X_npb_unique_dtp = spike_rates_patterned_behaviour_0p25[correlated_neuron_indices_unique_npb_dtp].transpose()
Y_npb_unique_dtp = distance_to_poke_non_patterned_behaviour_0p25


X_npb_unique_train = X_npb_unique_dtp[:700, :]
y_npb_unique_train = Y_npb_unique_dtp[:700]
X_npb_unique_test = X_npb_unique_dtp[700:, :]
y_npb_unique_test = Y_npb_unique_dtp[700:]


#    Linear
regressor_npb_unique_dtp = linear_model.LinearRegression(normalize=True)
regressor_npb_unique_dtp.fit(X_npb_unique_train, y_npb_unique_train)
print(regressor_npb_unique_dtp.score(X_npb_unique_train, y_npb_unique_train))
print(regressor_npb_unique_dtp.score(X_npb_unique_test, y_npb_unique_test))

Y_npb_unique_dtp_pred = regressor_npb_unique_dtp.predict(X_npb_unique_test)

#   Polynomial
model_npb_unique_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_npb_unique_dtp.fit(X_npb_unique_train, y_npb_unique_train)
print(model_npb_unique_dtp.score(X_npb_unique_train, y_npb_unique_train))
print(model_npb_unique_dtp.score(X_npb_unique_test, y_npb_unique_test))

Y_npb_unique_dtp_pred = model_npb_unique_dtp.predict(X_npb_unique_test)

plt.plot(Y_npb_unique_dtp_pred)
plt.plot(y_npb_unique_test)

#   Mixed X (firing rates) = patterned, Y (dtp) = non patterned

#    Linear
regressor_pb_npb_dtp = linear_model.LinearRegression(normalize=True)
regressor_pb_npb_dtp.fit(X_pb_unique_train, y_npb_unique_train)
print(regressor_pb_npb_dtp.score(X_pb_unique_train, y_npb_unique_train))
print(regressor_pb_npb_dtp.score(X_pb_unique_test, y_npb_unique_test))

Y_pb_npb_dtp_pred = regressor_pb_npb_dtp.predict(X_pb_unique_test)

#   Polynomial
model_pb_npb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_pb_npb_dtp.fit(X_pb_unique_train, y_npb_unique_train)
print(model_pb_npb_dtp.score(X_pb_unique_train, y_npb_unique_train))
print(model_pb_npb_dtp.score(X_pb_unique_test, y_npb_unique_test))

Y_pb_npb_dtp_pred = model_pb_npb_dtp.predict(X_pb_unique_test)

plt.plot(Y_pb_npb_dtp_pred)
plt.plot(y_npb_unique_test)

# </editor-fold>

# </editor-fold>
# -------------------------------------------------


