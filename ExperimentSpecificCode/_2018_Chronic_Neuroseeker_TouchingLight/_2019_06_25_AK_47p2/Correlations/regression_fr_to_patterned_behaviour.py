
from os.path import join
import numpy as np

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, regression_functions as reg_funcs

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics, pipeline, preprocessing

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
# <editor-fold desc="COMMON VARIABLES">

smoothing_time = 0.2  # The smoothing window for both the firing rates and the distance to poke time series
smoothing_frames = int(smoothing_time / 0.00833)

fr_final_smoothing_time = 1  # The final smoothing window of the firing rates
fr_extra_smoothing_frames = int(fr_final_smoothing_time / smoothing_time)

leave_percentage_out = 0.005

model = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model_npb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=None, fit_intercept=True))


common_pb_npb_dtp_neuron_indices = np.intersect1d(correlated_neuron_indices['pb_dtp'], correlated_neuron_indices['npb_dtp'])

correlated_neuron_indices_unique_pb_dtp = np.delete(correlated_neuron_indices['pb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['pb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))
correlated_neuron_indices_unique_npb_dtp = np.delete(correlated_neuron_indices['npb_dtp'],
                                                    np.argwhere(np.isin(correlated_neuron_indices['npb_dtp'],
                                                                        common_pb_npb_dtp_neuron_indices)))

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="REGRESSION OF DISTANCE TO POKE WITH FIRING RATES FOR PATTERNED BEHAVIOUR NEURONS AND EPOCHS">

# First smooth the distance to poke and the firing rates with a small window and then smooth only the firing rates
# further denoting the amount of past time a classifier needs to use


spike_rates_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates[:, windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames) * 0.00833 * smoothing_frames
distance_to_poke_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)


spike_rates_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates_patterned_behaviour_smoothed, np.mean, fr_extra_smoothing_frames, 1)
distance_to_poke_patterned_behaviour_smoothed = \
    np.array(distance_to_poke_patterned_behaviour_smoothed[len(distance_to_poke_patterned_behaviour_smoothed)
             - spike_rates_patterned_behaviour_smoothed.shape[1]:])

distance_to_poke_patterned_behaviour_smoothed = preprocessing.StandardScaler().\
    fit_transform(np.reshape(distance_to_poke_patterned_behaviour_smoothed, (-1, 1))).reshape(-1)

# Then run mutliple times the regression leaving out some samples every time
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices_unique_pb_dtp].transpose()
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices_unique_npb_dtp].transpose()
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices['pb_dtp']].transpose()
X_pb_dtp = spike_rates_patterned_behaviour_smoothed[correlated_neuron_indices['dtp']].transpose()
X_pb_dtp = spike_rates_patterned_behaviour_smoothed.transpose()

Y_pb_dtp = distance_to_poke_patterned_behaviour_smoothed

X_pb_dtp = preprocessing.StandardScaler().fit_transform(X_pb_dtp)

Y_pb_dtp_pred = reg_funcs.looped_regression(X_pb_dtp, Y_pb_dtp, model, leave_percentage_out)

np.save(join(regressions_folder, 'reg__leave1out_linear_2nd_order__dtp_pb_vs_fr_of_unique_npb_dtp_corr_neurons.npy'), Y_pb_dtp_pred)

plt.plot(Y_pb_dtp)
plt.plot(Y_pb_dtp_pred)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="REGRESSION OF DISTANCE TO POKE WITH FIRING RATES FOR NON PATTERNED BEHAVIOUR NEURONS AND EPOCHS">

# First smooth the distance to poke and the firing rates with a small window and then smooth only the firing rates
# further denoting the amount of past time a classifier needs to use

spike_rates_non_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates[:, windows_of_non_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)
distance_to_poke_non_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_non_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)

spike_rates_non_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates_non_patterned_behaviour_smoothed, np.mean, fr_extra_smoothing_frames, 1)
distance_to_poke_non_patterned_behaviour_smoothed = \
    np.array(distance_to_poke_non_patterned_behaviour_smoothed[len(distance_to_poke_non_patterned_behaviour_smoothed)
             - spike_rates_non_patterned_behaviour_smoothed.shape[1]:])

distance_to_poke_non_patterned_behaviour_smoothed = preprocessing.StandardScaler().\
    fit_transform(np.reshape(distance_to_poke_non_patterned_behaviour_smoothed, (-1, 1))).reshape(-1)

# Then run mutliple times the regression leaving out some samples every time
X_npb_dtp = spike_rates_non_patterned_behaviour_smoothed[correlated_neuron_indices_unique_npb_dtp].transpose()
X_npb_dtp = spike_rates_non_patterned_behaviour_smoothed[correlated_neuron_indices_unique_pb_dtp].transpose()
X_npb_dtp = spike_rates_non_patterned_behaviour_smoothed[correlated_neuron_indices['npb_dtp']].transpose()
X_npb_dtp = spike_rates_non_patterned_behaviour_smoothed[correlated_neuron_indices['dtp']].transpose()

Y_npb_dtp = distance_to_poke_non_patterned_behaviour_smoothed

X_npb_dtp = preprocessing.StandardScaler().fit_transform(X_npb_dtp)

Y_npb_dtp_pred = reg_funcs.looped_regression(X_npb_dtp, Y_npb_dtp, model, leave_percentage_out)
np.save(join(regressions_folder, 'reg__leave1out_linear_2nd_order__dtp_npb_vs_fr_of_unique_pb_dtp_corr_neurons.npy'), Y_npb_dtp_pred)


plt.plot(Y_npb_dtp)
plt.plot(Y_npb_dtp_pred)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="FEATURE SELECTION OF DISTANCE TO POKE WITH FIRING RATES FOR PATTERNED BEHAVIOUR NEURONS AND EPOCHS">


smoothing_time = 0.25
smoothing_frames = int(smoothing_time / 0.00833)

distance_to_poke_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames)

spike_counts_patterned_behaviour_smoothed = \
    binning.rolling_window_with_step(spike_rates[:, windows_of_patterned_behaviour],
                                     np.mean, smoothing_frames, smoothing_frames) * 0.00833 * smoothing_frames

spike_counts_patterned_behaviour_smoothed = np.ceil(spike_counts_patterned_behaviour_smoothed).astype(np.int)

distance_to_poke_patterned_behaviour_smoothed = preprocessing.StandardScaler().\
    fit_transform(np.reshape(distance_to_poke_patterned_behaviour_smoothed, (-1, 1))).reshape(-1)


high_mi_dtp_neurons = np.squeeze(np.argwhere(mis['dtp'] > 0.01))
high_mi_dtp_neurons = correlated_neuron_indices['dtp']

X = spike_counts_patterned_behaviour_smoothed[high_mi_dtp_neurons, :700].transpose()
Y = distance_to_poke_patterned_behaviour_smoothed[:700]
Y -= Y.min()

'''
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LassoCV(cv=5, n_alphas=100, fit_intercept=True,
                                                                                  verbose=2, n_jobs=10))
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.Lasso(alpha=0.04, fit_intercept=True))
model_pb_dtp = pipeline.make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression(fit_intercept=True))
model_pb_dtp.fit(X, Y)

selector = RFE(model_pb_dtp, 500, verbose=2)
selector = selector.fit(X_poly, Y)
'''

X_poly_model = PolynomialFeatures(2)
X_poly = X_poly_model.fit_transform(X)

model_pb_dtp = linear_model.LinearRegression(fit_intercept=True)

model_pb_dtp = linear_model.Lasso(alpha=0.02, fit_intercept=True, max_iter=10000, normalize=False)

model_pb_dtp = linear_model.Ridge(alpha=1, solver='saga', fit_intercept=True, max_iter=10000, normalize=True)

model_pb_dtp.fit(X_poly, Y)


plt.plot(Y)
plt.plot(model_pb_dtp.predict(X_poly))

test_range = np.squeeze(np.concatenate((np.arange(500), np.arange(1500, 2120))))

test_range = np.arange(700, spike_counts_patterned_behaviour_smoothed.shape[1])
Y_test = np.array(distance_to_poke_patterned_behaviour_smoothed)[test_range]
#Y_test -= Y_test.min()
X_test = spike_counts_patterned_behaviour_smoothed[high_mi_dtp_neurons, :]
X_test = X_test[:, test_range].transpose()

X_test = X_poly_model.fit_transform(X_test)

plt.plot(Y_test)
plt.plot(model_pb_dtp.predict(X_test))
plt.plot(binning.rolling_window_with_step(model_pb_dtp.predict(X_test), np.mean, 10, 1))

model_pb_dtp.score(X_test, Y_test)
metrics.r2_score(Y_test[:-10], binning.rolling_window_with_step(model_pb_dtp.predict(X_test), np.mean, 10, 1))

alpha = 0.0187 # for 90 neurons Score = 0.5141
alpha = 0.0269 # for 234 neurons Score = 0.538
alpha = 0.0269 # for 336 neuorns (dtp) Score = 0.569


'''
# Testing to see if linear lasso regression with extra past time points would help. It doesn't
past_steps = 5

spike_counts_patterned_behaviour_smoothed_past_steps = \
    reg_funcs.add_past_onto_X(spike_counts_patterned_behaviour_smoothed, past_steps)

X = spike_counts_patterned_behaviour_smoothed_past_steps.transpose()
Y = distance_to_poke_patterned_behaviour_smoothed

lasso_regressor = linear_model.LassoCV(cv=None, fit_intercept=True, verbose=2, n_jobs=10)
model_pb_dtp = SelectFromModel(lasso_regressor, prefit=False)
model_pb_dtp.fit(X, Y)

model_pb_dtp.score(X, Y)
plt.plot(Y)
plt.plot(model_pb_dtp.predict(X))

all_past_models = []
for ps in np.arange(past_steps):
    spike_counts_patterned_behaviour_smoothed_past_steps = \
        reg_funcs.add_past_onto_X(spike_counts_patterned_behaviour_smoothed, ps + 1)

    X = spike_counts_patterned_behaviour_smoothed_past_steps.transpose()
    Y = distance_to_poke_patterned_behaviour_smoothed
    print('Doing past step {}'.format(str(ps)))

    lasso_regressor = linear_model.LassoCV(cv=None, fit_intercept=True, verbose=2, n_jobs=10)
    model_pb_dtp = SelectFromModel(lasso_regressor, prefit=True)
    model_pb_dtp.fit(X, Y)

    all_past_models.append(model_pb_dtp)

lasso_regressor = linear_model.LassoCV(cv=None, fit_intercept=True, verbose=0, n_jobs=10)
lasso_models = []
lasso_predictions = []
for ps in np.arange(past_steps):
    X_fs = all_past_models[ps].transform(
        reg_funcs.add_past_onto_X(spike_counts_patterned_behaviour_smoothed, ps + 1).transpose())
    lasso_models.append(lasso_regressor.fit(X_fs, Y))
    lasso_predictions.append(lasso_regressor.predict(X_fs))
'''

spike_counts_full_smoothed = \
    binning.rolling_window_with_step(spike_rates, np.mean, smoothing_frames, smoothing_frames) * 0.00833 * smoothing_frames
spike_counts_full_smoothed = np.ceil(spike_counts_full_smoothed).astype(np.int)

X_full = spike_counts_full_smoothed[high_mi_dtp_neurons, :].transpose()
Y_full = binning.rolling_window_with_step(distances_rat_to_poke_all_frames,
                                          np.mean, smoothing_frames, smoothing_frames)
Y_full = preprocessing.StandardScaler().fit_transform(np.reshape(Y_full, (-1, 1))).reshape(-1)

X_full_poly = X_poly_model.fit_transform(X_full)
np.save(join(regressions_folder, 'X_full_time_0p1s_smooth_polynomial_features.npy'), X_full_poly)

X_full_poly = np.load(join(regressions_folder, 'X_full_time_0p1s_smooth_polynomial_features.npy'))

Y_full_pred = model_pb_dtp.predict(X_full_poly)

metrics.r2_score(Y_full, Y_full_pred)

plt.plot(Y_full)
plt.plot(Y_full_pred)

distance = np.sqrt(np.power(Y_full - Y_full_pred, 2))
distance_smoothed = binning.rolling_window_with_step(distance, np.mean, 50, 1)
plt.plot(distance_smoothed)
# </editor-fold>
# -------------------------------------------------