
from os.path import join
import numpy as np
import pickle
import bisect

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp

from sklearn import linear_model, metrics, model_selection, preprocessing
from scipy import stats
import statsmodels.stats.api as sms

from npeet.lnc import MI

import pygam as pg

import sequence_viewer as sv
import transform as tr
import drop_down as dd
import slider as sl

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

ballistic_mov_folder = join(results_folder, 'EventsCorrelations', 'StartBallisticMovToPoke')

# Load data
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)

distances_rat_head_to_poke_smoothed = np.load(join(regressions_folder, 'distances_rat_head_to_poke_smoothed.npy'))

frame_regions_away_from_poke = np.load(join(regressions_folder, 'frame_regions_away_from_poke.npy'))

distance_to_travel_to_poke = np.load(join(regressions_folder, 'distance_to_travel_to_poke.npy'))

distance_to_travel_to_poke_in_periods = np.load(join(regressions_folder, 'distance_to_travel_to_poke_in_periods.npy'),
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

distance_to_travel_to_poke_smoothed = binning.rolling_window_with_step(distance_to_travel_to_poke,
                                                              np.mean, fr_smooth_frames, fr_smooth_frames)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="DO THE MIs (RUN ONCE)">
# NOTE The MIs were done with fr_smooth_time = 0.5
n = 0
mi_spike_rates_vs_distance_to_travel_to_poke = []
for rate in spike_rates_away_from_poke_smoothed:
    mi_spike_rates_vs_distance_to_travel_to_poke.append(MI.mi_LNC([rate.tolist(), distance_to_travel_to_poke],
                                                                  k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spike_rates_vs_distance_to_travel_to_poke = np.array(mi_spike_rates_vs_distance_to_travel_to_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_travel_to_poke.npy'),
        mi_spike_rates_vs_distance_to_travel_to_poke)


#   Do the shuffle of the best fr vs distance to travel to next poke
max_neuron = np.argmax(mi_spike_rates_vs_distance_to_travel_to_poke)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_away_from_poke_smoothed[max_neuron],
                                                 distance_to_travel_to_poke,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder,
             'shuffled_mut_info_spike_rate_{}_vs_distance_to_travel_to_poke.npy'.format(str(max_neuron))),
        shuffled)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS OF THE MI CALCULATIONS">

#   Load the MIs and the shuffle
distance_to_travel_to_poke_smoothed = np.array(distance_to_travel_to_poke_smoothed)

mi_spike_rates_vs_distance_to_travel_to_poke = \
    np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_travel_to_poke.npy'))
max_neuron = np.argmax(mi_spike_rates_vs_distance_to_travel_to_poke)
shuffled = np.load(join(mutual_information_folder,
                        'shuffled_mut_info_spike_rate_{}_vs_distance_to_travel_to_poke.npy'.
                        format(str(max_neuron))))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]

#   Have a look at the MIs vs the chance hist
#       Log
_ = plt.hist(mi_spike_rates_vs_distance_to_travel_to_poke, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
_ = plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(0, 0, 1, 1))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
#       Linear
_ = plt.hist(mi_spike_rates_vs_distance_to_travel_to_poke, bins= 200, color=(0, 0, 1, 1))
_ = plt.hist(shuffled, bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, confi_intervals[0], confi_intervals[1]], 0, 20)

#   Where are the neurons
high_corr_neurons_index = np.squeeze(np.argwhere(mi_spike_rates_vs_distance_to_travel_to_poke > confi_intervals[1]))
high_corr_neurons = template_info.loc[high_corr_neurons_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=high_corr_neurons,
                                     dot_sizes=mi_spike_rates_vs_distance_to_travel_to_poke[high_corr_neurons_index] * 4000,
                                     font_size=5)

#   Look at individual spike rates vs the distance traveled
hipp_neuron_index = template_info[template_info['template number'] == 89].index.values[0]
neuron_index_to_show = hipp_neuron_index
neuron_index_to_show = max_neuron
plt.plot((distance_to_travel_to_poke_smoothed - distance_to_travel_to_poke_smoothed.min()) /
         (distance_to_travel_to_poke_smoothed.max() - distance_to_travel_to_poke_smoothed.min()))
plt.plot((spike_rates_away_from_poke_smoothed[neuron_index_to_show] - spike_rates_away_from_poke_smoothed[neuron_index_to_show].min()) /
         (spike_rates_away_from_poke_smoothed[neuron_index_to_show].max() - spike_rates_away_from_poke_smoothed[neuron_index_to_show].min()))

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SET UP THE REGRESSION DATA">

#   Use only a few seconds as the rat approaches the poke
distance_to_travel_to_poke_smoothed = np.array(distance_to_travel_to_poke_smoothed)

period_duration_to_select_over = 5  # Use this amount of seconds before the poke
min_period_duration_to_keep = 3  # Ignore periods shorter than this
period_duration_to_select_over_in_frames = period_duration_to_select_over * 120
min_period_duration_to_keep_in_frames = min_period_duration_to_keep * 120

frame_windows_away_from_poke_arriving_to_poke = np.empty(0)
distance_to_travel_to_poke_close_to_poke = np.empty(0)
starts_of_periods = [0]
lengths_of_periods = []
for w in np.arange(len(frame_windows_away_from_poke_in_periods)):
    frames = np.array(frame_windows_away_from_poke_in_periods[w])
    distances = np.array(distance_to_travel_to_poke_in_periods[w])
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
mi_spike_rates_vs_distance_to_travel_to_poke = \
    np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_travel_to_poke.npy'))
max_neuron = np.argmax(mi_spike_rates_vs_distance_to_travel_to_poke)
shuffled = np.load(join(mutual_information_folder,
                        'shuffled_mut_info_spike_rate_{}_vs_distance_to_travel_to_poke.npy'.
                        format(str(max_neuron))))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]

high_corr_neurons_index = np.squeeze(np.argwhere(mi_spike_rates_vs_distance_to_travel_to_poke > confi_intervals[1]))

#   Standartize X, Y
Y = preprocessing.StandardScaler().\
    fit_transform(np.reshape(distance_to_travel_to_poke_close_to_poke_smoothed, (-1, 1))).reshape(-1)

X = preprocessing.StandardScaler().\
    fit_transform(spike_rates_away_from_poke_close_to_poke_smoothed[high_corr_neurons_index].transpose())

#   Or not
Y = np.array(distance_to_travel_to_poke_close_to_poke_smoothed)
X = spike_rates_away_from_poke_close_to_poke_smoothed[high_corr_neurons_index].transpose()


#   Set up the polynomial
X_poly_model = preprocessing.PolynomialFeatures(2)
X_poly = X_poly_model.fit_transform(X)

#   Set up the regressors
model_linear = linear_model.LinearRegression(fit_intercept=True)
model_lassoCV = linear_model.LassoCV(cv=5, fit_intercept=True)
model_lasso = linear_model.Lasso(alpha=0.02, fit_intercept=True, max_iter=10000, normalize=False)

model_gam = pg.LinearGAM()

Y = np.exp(Y) / (np.exp(Y) + 1)
model_gam = pg.GammaGAM()

#   Split the data to train and test sets
#       Spit randomly by points
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)


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


#   Run the regression

model = model_lasso
model.fit(X_train, Y_train)

# ------GAM stuff
# Transform Y for log link functions

# Fit
model = model_gam
model.gridsearch(X_train, Y_train)

pickle.dump(model, open(join(regressions_folder, 'gamma_GAM_frs_095_vs_distance_to_travel_to_poke.pcl'), "wb"))
model = pickle.load( open(join(regressions_folder, 'linear_GAM_frs_095_vs_distance_to_travel_to_poke.pcl'), "rb"))

def logprob_to_y(logprob):
    return np.log(logprob / (1 - logprob))

# Show partial dependence of each neuron
signif_thres = 1e-3
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

# Do a lasso only with the significant neurons from the GAM
X_train = X_used[train_periods, :]
X_train = X_train[:, sig_neurons_index]
Y_train = Y[train_periods]
X_test = X_used[test_periods, :]
X_test = X_test[:, sig_neurons_index]
Y_test = Y[test_periods]

model = model_lasso
model.fit(X_train, Y_train)
# ------

plt.plot(Y_test)
plt.plot(model.predict(X_test))

plt.plot(Y_train)
plt.plot(model.predict(X_train))

print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))

plt.plot(logprob_to_y(Y_test))
plt.plot(logprob_to_y(model.predict(X_test)))

plt.plot(logprob_to_y(Y_train))
plt.plot(logprob_to_y(model.predict(X_train)))

plt.plot(Y)
plt.plot(model.predict(X_used))

start_frame_of_each_train_period = frame_windows_away_from_poke_arriving_to_poke[starts_of_periods[train_periods_indices]]
start_frame_of_each_test_period = frame_windows_away_from_poke_arriving_to_poke[starts_of_periods[test_period_indices]]

#   Fit all data and score each period
model_full = model_gam
model_full.gridsearch(X_used, Y)
pickle.dump(model_full, open(join(regressions_folder, 'linear_GAM_full_frs_099_vs_distance_to_travel_to_poke.pcl'), "wb"))

model_full = pickle.load(open(join(regressions_folder, 'linear_GAM_full_frs_099_vs_distance_to_travel_to_poke.pcl'), "rb"))

Y_pred = model_full.predict(X_used)

scores_test_periods = []
score_test_periods_cont = np.ones(len(Y)) * 0.1
for i in np.arange(len(starts_of_periods_smoothed)):
    minus = int(1 * 120 / fr_smooth_frames)
    x = X_used[starts_of_periods_smoothed[i] + minus:starts_of_periods_smoothed[i]+lengths_of_periods_smoothed[i], :]
    y = Y[starts_of_periods_smoothed[i] + minus:starts_of_periods_smoothed[i]+lengths_of_periods_smoothed[i]]
    y_pred = model_full.predict(x)
    if np.any(np.isnan(y_pred)):
        index = np.squeeze(np.argwhere(np.isnan(y_pred)))
        y_pred[index] = y_pred[index-1]
    r2 = metrics.mean_squared_error(y, y_pred)
    if r2 < 0.1:
        r2 = 0.1
    scores_test_periods.append(r2)
    score_test_periods_cont[starts_of_periods_smoothed[i]:starts_of_periods_smoothed[i]+lengths_of_periods_smoothed[i]] = \
        scores_test_periods[-1]

scores_test_periods = np.array(scores_test_periods)
scores_test_periods_cont_norm = (score_test_periods_cont - score_test_periods_cont.min()) / \
                           (score_test_periods_cont.max() - score_test_periods_cont.min()) * 2 +1
scores_test_periods_norm = (scores_test_periods - scores_test_periods.min()) / (scores_test_periods.max() - scores_test_periods.min()) * 2 +1

window_step = 10
t = []
for i in np.arange(len(Y)):
    x = X_used[i:i+window_step, :]
    y = Y[i:i+window_step]
    y_pred = model_full.predict(x)
    if np.any(np.isnan(y_pred)):
        index = np.squeeze(np.argwhere(np.isnan(y_pred)))
        y_pred[index] = y_pred[index-1]
    t.append(metrics.mean_squared_error(y, y_pred))
scores_test_cont = np.zeros(len(t))
scores_test_cont[window_step:] = t[:-window_step]


plt.plot(Y)
plt.plot(Y_pred)
plt.plot(scores_test_periods_cont_norm)
plt.hlines(y=1.7, xmin=0, xmax=len(Y))
plt.plot(scores_test_cont)

plt.plot(logprob_to_y(Y))
plt.plot(logprob_to_y(Y_pred))
plt.plot(scores_test_periods_cont_norm)
plt.hlines(y=1.35, xmin=0, xmax=len(Y))


# <editor-fold desc="HAVE A LOOK AT THE DIFFERENT PERIODS">

#   Average tracked and non tracked periods
r2_norm_threshold = 1.55
length_of_period_smoothed = (int(period_duration_to_select_over * 120 / fr_smooth_frames))
tracked_Y_over_period = []
not_tracked_Y_over_period = []
Y_over_period = []
tracked_Y_pred_over_period = []
not_tracked_Y_pred_over_period = []
Y_pred_over_period = []

num_of_tracked = 0
num_of_not_tracked = 0

for p in np.arange(len(starts_of_periods)):
    if lengths_of_periods_smoothed[p] == lengths_of_periods_smoothed.max():
        start_point = starts_of_periods_smoothed[p]
        end_point = starts_of_periods_smoothed[p] + lengths_of_periods_smoothed[p]

        period_Y = Y[start_point:end_point]
        period_Y_pred = Y_pred[start_point:end_point]

        Y_over_period.append(period_Y)
        Y_pred_over_period.append(period_Y_pred)

        if scores_test_periods_norm[p] < r2_norm_threshold:
            tracked_Y_over_period.append(period_Y)
            tracked_Y_pred_over_period.append(period_Y_pred)
            num_of_tracked += 1
        else:
            not_tracked_Y_over_period.append(period_Y)
            not_tracked_Y_pred_over_period.append(period_Y_pred)
            num_of_not_tracked += 1

tracked_Y_over_period = np.array(tracked_Y_over_period)
tracked_Y_pred_over_period = np.array(tracked_Y_pred_over_period)
not_tracked_Y_over_period = np.array(not_tracked_Y_over_period)
not_tracked_Y_pred_over_period = np.array(not_tracked_Y_pred_over_period)

std_tracked_Y_over_period = np.nanstd(tracked_Y_over_period, 0)
std_tracked_Y_pred_over_period = np.nanstd(tracked_Y_pred_over_period, 0)
std_not_tracked_Y_over_period = np.nanstd(not_tracked_Y_over_period, 0)
std_not_tracked_Y_pred_over_period = np.nanstd(not_tracked_Y_pred_over_period, 0)

avg_tracked_Y_over_period = np.nanmean(tracked_Y_over_period, 0)
avg_tracked_Y_pred_over_period = np.nanmean(tracked_Y_pred_over_period, 0)
avg_not_tracked_Y_over_period = np.nanmean(not_tracked_Y_over_period, 0)
avg_not_tracked_Y_pred_over_period = np.nanmean(not_tracked_Y_pred_over_period, 0)


f = plt.figure(0)
a1 = f.add_subplot(1,2,1)
a1.plot(avg_tracked_Y_over_period)
a1.plot(sms.DescrStatsW(tracked_Y_over_period).tconfint_mean(0.01)[0],ls='--')
a1.plot(sms.DescrStatsW(tracked_Y_over_period).tconfint_mean(0.01)[1], ls='--')
a1.plot(avg_tracked_Y_pred_over_period)
a1.plot(sms.DescrStatsW(tracked_Y_pred_over_period).tconfint_mean(0.01)[0],  ls='--')
a1.plot(sms.DescrStatsW(tracked_Y_pred_over_period).tconfint_mean(0.01)[1],  ls='--')

a2 = f.add_subplot(1,2,2)
a2 .plot(avg_not_tracked_Y_over_period)
a2.plot(sms.DescrStatsW(not_tracked_Y_over_period).tconfint_mean(0.01)[0],ls='--')
a2.plot(sms.DescrStatsW(not_tracked_Y_over_period).tconfint_mean(0.01)[1], ls='--')
a2 .plot(avg_not_tracked_Y_pred_over_period)
a2.plot(sms.DescrStatsW(not_tracked_Y_pred_over_period).tconfint_mean(0.01)[0],  ls='--')
a2.plot(sms.DescrStatsW(not_tracked_Y_pred_over_period).tconfint_mean(0.01)[1],  ls='--')

#   Make live video showing where the periods are tracked
Y_pred = model_full.predict(X_used)
frame = 0
frame_starts_of_periods = frame_windows_away_from_poke_arriving_to_poke[starts_of_periods]
is_period_tracked = False


def is_frame_tracked(fr):
    result = False
    if np.any(np.isin(frame_windows_away_from_poke_arriving_to_poke, fr)):
        period = bisect.bisect(frame_windows_away_from_poke_arriving_to_poke[starts_of_periods], fr) - 1
        if scores_test_periods_norm[period] < r2_norm_threshold:
            result = True
    return result


def show_distances(fr, ax):
    step = 20
    fr_in_afp = np.argwhere(frame_windows_away_from_poke_arriving_to_poke == fr)
    if len(fr_in_afp) > 0:
        ax.clear()
        fr_in_afp = np.squeeze(fr_in_afp)
        fr_smooth = int(fr_in_afp / fr_smooth_frames)
        ax.plot(Y[fr_smooth - step: fr_smooth + step])
        ax.plot(Y_pred[fr_smooth - step: fr_smooth + step])
        ax.plot(scores_test_periods_cont_norm[fr_smooth - step: fr_smooth + step])
        ax.vlines(x=step, ymin=-1.2, ymax=3)
        ax.hlines(y=r2_norm_threshold, xmin=0, xmax=2*step)


fig = plt.figure(3)
ax = fig.add_subplot(111)
args = [ax]
out_tr = None

sv.image_sequence(globals(), 'frame', 'video_file')
dd.connect_repl_var(globals(), 'frame_starts_of_periods', 'frame')
tr.connect_repl_var(globals(), 'frame', 'is_period_tracked', 'is_frame_tracked')
tr.connect_repl_var(globals(), 'frame', 'out_tr', 'show_distances', 'args')
# </editor-fold>

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="DO THE MIs BETWEEN THE SPIKE RATES AND THE QUALITY OF PREDICTION OF THE DISTANCE TO TRAVEL TO POKE (RUN ONCE)">

n = 0
mi_spike_rates_vs_quality_of_regression_to_travel_to_poke = []
for rate in spike_rates_away_from_poke_close_to_poke_smoothed:
    mi_spike_rates_vs_quality_of_regression_to_travel_to_poke.append(MI.mi_LNC([rate.tolist(), list(scores_test_cont)],
                                                                  k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spike_rates_vs_quality_of_regression_to_travel_to_poke = np.array(mi_spike_rates_vs_quality_of_regression_to_travel_to_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spike_rates_vs_quality_of_regression_to_travel_to_poke.npy'),
        mi_spike_rates_vs_quality_of_regression_to_travel_to_poke)


#   Visualise
_ = plt.hist(mi_spike_rates_vs_quality_of_regression_to_travel_to_poke, bins= 200, color=(0, 0, 1, 1))

max_neurons = np.squeeze(np.argwhere(np.array(mi_spike_rates_vs_quality_of_regression_to_travel_to_poke) > 0.016))
plt.plot((scores_test_cont - scores_test_cont.min()) /
         (scores_test_cont.max() - scores_test_cont.min()))
max_neuron = max_neurons[0]
plt.plot((spike_rates_away_from_poke_close_to_poke_smoothed[max_neuron] - spike_rates_away_from_poke_close_to_poke_smoothed[max_neuron].min()) /
         (spike_rates_away_from_poke_close_to_poke_smoothed[max_neuron].max() - spike_rates_away_from_poke_close_to_poke_smoothed[max_neuron].min()))

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="FIND THE FRAMES WHERE THE RAT STARTS A BALLISTIC MOVE TOWARDS THE POKE AND CHECK NEURONS THAT CHANGE ACTIVITY AROUND THESE">

#   Create the events manually
register = False
states = [True, False]
def do_nothing(input):
    return input

dd.connect_repl_var(globals(), 'states', 'register', 'do_nothing')


start_of_ballistic_traj_frames_hand_picked = []

def on_pick(ev):
    if register:
        start_of_ballistic_traj_frames_hand_picked.append(ev.xdata)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.canvas.mpl_connect('button_press_event', on_pick)
ax.plot(distance_to_travel_to_poke)


start_of_ballistic_traj_frames_hand_picked = np.array(start_of_ballistic_traj_frames_hand_picked).astype(np.int)
np.save(join(ballistic_mov_folder, 'start_of_ballistic_traj_frames_hand_picked.npy'),
        start_of_ballistic_traj_frames_hand_picked)

#   Have a look
start_of_ballistic_traj_frames_hand_picked = np.load(join(ballistic_mov_folder, 'start_of_ballistic_traj_frames_hand_picked.npy'))
plt.plot(distance_to_travel_to_poke)
plt.vlines(x=start_of_ballistic_traj_frames_hand_picked, ymin=0, ymax=4000)

# ------
#   Find any increasing or decreasing neurons
start_of_ballistic_traj_frames_hand_picked = frame_windows_away_from_poke[start_of_ballistic_traj_frames_hand_picked]
start_of_ballistic_traj_time_points = ev_video['AmpTimePoints'].iloc[start_of_ballistic_traj_frames_hand_picked].values


time_around_start_of_bal_mov = 5
frames_around_start_of_bal_mov = time_around_start_of_bal_mov * 120

avg_firing_rate_around_start_bal_mov = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=start_of_ballistic_traj_time_points,
                                                                                ev_video_df=ev_video,
                                                                                time_around_event=time_around_start_of_bal_mov)

decreasing_firing_rates_neuron_index, decreasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_start_bal_mov,
                                                           time_around_pattern=time_around_start_of_bal_mov,
                                                           pattern_regions_to_compare=[0, 0.6, 0.8, 1.2],
                                                           comparison_factor=3, comparison_direction='decrease',
                                                           baserate=0.15)

fr_funcs.show_firing_rates_around_event(decreasing_firing_rates)

template_info_decreasing_fr_neurons = template_info.iloc[decreasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_decreasing_fr_neurons)

pd.to_pickle(template_info_decreasing_fr_neurons, join(ballistic_mov_folder, 'ti_decreasing_neurons_on_start_ballistic.npy'))


increasing_firing_rates_neuron_index, increasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_start_bal_mov,
                                                           time_around_pattern=time_around_start_of_bal_mov,
                                                           pattern_regions_to_compare=[0, 0.6, 0.8, 1.2],
                                                           comparison_factor=3, comparison_direction='increase',
                                                           baserate=0.1)
fr_funcs.show_firing_rates_around_event(increasing_firing_rates)

template_info_increasing_fr_neurons = template_info.iloc[increasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_increasing_fr_neurons)

pd.to_pickle(template_info_increasing_fr_neurons, join(ballistic_mov_folder, 'ti_increasing_neurons_on_start_ballistic.npy'))


# ------
# Have a look at the individual raster plots of the neurons with the largest change
frs = decreasing_firing_rates_neuron_rand_index  # decreasing_firing_rates_neuron_index or increasing_firing_rates_neuron_index
index = 0
fig1 = plt.figure(0)
fig2 = plt.figure(1)
output = None
args = [frs, avg_firing_rate_around_start_bal_mov, template_info, spike_info,
        start_of_ballistic_traj_time_points, frames_around_start_of_bal_mov, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(frs) - 1])

# -----
# Randomize to test robustness

start_of_ballistic_traj_time_points_rand = start_of_ballistic_traj_time_points + \
                                           ((np.random.choice([1], len(start_of_ballistic_traj_time_points))) * 20000).astype(np.int)


avg_firing_rate_around_start_bal_mov_rand = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=start_of_ballistic_traj_time_points_rand,
                                                                                ev_video_df=ev_video,
                                                                                time_around_event=time_around_start_of_bal_mov)

decreasing_firing_rates_neuron_rand_index, decreasing_firing_rates_rand = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_start_bal_mov_rand,
                                                           time_around_pattern=time_around_start_of_bal_mov,
                                                           pattern_regions_to_compare=[0, 0.7, 0.8, 1.1],
                                                           comparison_factor=3, comparison_direction='decrease',
                                                           baserate=0.15)
fr_funcs.show_firing_rates_around_event(decreasing_firing_rates_rand)

# </editor-fold>
# -------------------------------------------------