


from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis import binning
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import random_projection

import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')

events_folder = join(data_folder, "events")

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

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
time_points_of_trial_pokes = np.load(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy'))
time_points_of_non_trial_pokes = np.load(join(events_definitions_folder,
                                              'events_first_pokes_after_5_delay_non_reward.npy'))
time_points_of_touch_ball = np.load(join(poke_folder, 'time_points_of_touch_ball.npy'))

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates_per_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(spike_rates_per_frame_filename)
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="GET THE SPIKE RATES OF THE RELEVANT NEURONS">
spike_rates_increase_on_trial = spike_rates[ti_increasing_neurons_on_trial_pokes.index.values, :]
spike_rates_decrease_on_trial = spike_rates[ti_decreasing_neurons_on_trial_pokes.index.values, :]
spike_rates_increase_on_non_trial = spike_rates[ti_increasing_neurons_on_non_trial_pokes.index.values, :]
spike_rates_decrease_on_non_trial = spike_rates[ti_decreasing_neurons_on_non_trial_pokes.index.values, :]

# plt.plot(np.transpose(binning.rolling_window_with_step(spike_rates_increase_on_trial, np.mean, 10, 10)))
# plt.vlines(x=time_points_of_trial_pokes/(const.SAMPLING_FREQUENCY * 0.0083*10), ymin=0, ymax=200)

all_neuron_templates = np.zeros(len(ti_increasing_neurons_on_trial_pokes) + len(ti_decreasing_neurons_on_trial_pokes)+
                                len(ti_increasing_neurons_on_non_trial_pokes) +
                                len(ti_decreasing_neurons_on_non_trial_pokes))
start = 0
end = len(ti_increasing_neurons_on_trial_pokes)
all_neuron_templates[start: end] = ti_increasing_neurons_on_trial_pokes['template number'].values

start += len(ti_increasing_neurons_on_trial_pokes)
end += len(ti_decreasing_neurons_on_trial_pokes)
all_neuron_templates[start: end] = ti_decreasing_neurons_on_trial_pokes['template number'].values

start += len(ti_decreasing_neurons_on_trial_pokes)
end += len(ti_increasing_neurons_on_non_trial_pokes)
all_neuron_templates[start: end] = ti_increasing_neurons_on_non_trial_pokes['template number'].values

start += len(ti_increasing_neurons_on_non_trial_pokes)
end += len(ti_decreasing_neurons_on_non_trial_pokes)
all_neuron_templates[start: end] = ti_decreasing_neurons_on_non_trial_pokes['template number'].values

all_neuron_templates = np.unique(all_neuron_templates)

# sr prefix mean the spike rates of all relevant neurons
sr = spike_rates[template_info[np.isin(template_info['template number'], all_neuron_templates)].
                                      index.values, :]

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="PCA SPIKE RATES">

# Get the frames around the events
window_size_seconds = 5
window_size_frames = int(window_size_seconds / 0.008333)
trial_frames = sync_funcs.time_point_to_frame_from_video_df(event_dataframes['ev_video'], time_points_of_trial_pokes)
non_trial_frames = sync_funcs.time_point_to_frame_from_video_df(event_dataframes['ev_video'], time_points_of_non_trial_pokes)

number_of_trial_pokes = len(trial_frames)
number_of_non_trial_pokes = len(non_trial_frames)

trial_windows_in_frames = [np.arange(trial_frame - window_size_frames, trial_frame + window_size_frames, 1) for
                           trial_frame in trial_frames]
non_trial_windows_in_frames = [np.arange(non_trial_frame - window_size_frames, non_trial_frame + window_size_frames, 1) for
                               non_trial_frame in non_trial_frames]

# Get random frames away from the events to use as baseline data
all_non_baseline_frames = np.concatenate((trial_windows_in_frames, non_trial_windows_in_frames))
number_of_pokes = all_non_baseline_frames.shape[0]
all_non_baseline_frames = all_non_baseline_frames.flatten()
total_number_of_frames = len(event_dataframes['ev_video']) - window_size_frames
baseline_frames = np.random.choice(np.delete(np.arange(total_number_of_frames), all_non_baseline_frames),
                                   number_of_pokes)

baseline_trial_windows_in_frames = [np.arange(baseline_trial_frame - window_size_frames, baseline_trial_frame + window_size_frames, 1) for
                                    baseline_trial_frame in baseline_frames[:len(trial_windows_in_frames)]]
baseline_non_trial_windows_in_frames = [np.arange(baseline_non_trial_frame - window_size_frames, baseline_non_trial_frame + window_size_frames, 1) for
                                        baseline_non_trial_frame in baseline_frames[len(trial_windows_in_frames):]]

'''
# Do PCA on raw data and then average this around the events
pca = PCA()
_ = pca.fit_transform(sr)
sr_pca = pca.components_

sr_all_neurons_pca_average_around_trials = sr_pca[:, trial_windows_in_frames].mean(1)
sr_all_neurons_pca_average_around_non_trials = sr_pca[:, non_trial_windows_in_frames].mean(1)


# Average spike rates across the events and then PCA the averages
sr_average_around_trials = sr[:, trial_windows_in_frames].mean(1)
sr_average_around_non_trials = sr[:, non_trial_windows_in_frames].mean(1)

pca_trials = PCA()
_ = pca_trials.fit_transform(sr_average_around_trials)
pca_all_neurons_avg_around_trials = pca_trials.components_

pca_non_trials = PCA()
_ = pca_non_trials.fit_transform(sr_average_around_non_trials)
pca_all_neurons_avg_around_non_trials = pca_non_trials.components_
'''

# Cut the trials
sr_trials = sr[:, trial_windows_in_frames]
sr_non_trials = sr[:, non_trial_windows_in_frames]

sr_baseline_trials = sr[:, baseline_trial_windows_in_frames]
sr_baseline_non_trials = sr[:, baseline_non_trial_windows_in_frames]

# Smooth
smoothing_window = int(0.25 * 120)  # in frames
step_window = 10  # in frames

sr_trials_smoothed = []
sr_baseline_trials_smoothed = []

for trial in np.arange(number_of_trial_pokes):
    sr_trials_smoothed.append(
        binning.rolling_window_with_step(sr_trials[:, trial, :],
                                         np.mean, smoothing_window, step_window))

    sr_baseline_trials_smoothed.append(
        binning.rolling_window_with_step(sr_baseline_trials[:, trial, :],
                                         np.mean, smoothing_window, step_window))

sr_non_trials_smoothed = []
sr_baseline_non_trials_smoothed = []

for non_trial in np.arange(number_of_non_trial_pokes):
    sr_non_trials_smoothed.append(
        binning.rolling_window_with_step(sr_non_trials[:, non_trial, :],
                                         np.mean, smoothing_window, step_window))

    sr_baseline_non_trials_smoothed.append(
        binning.rolling_window_with_step(sr_baseline_non_trials[:, non_trial, :],
                                         np.mean, smoothing_window, step_window))

sr_trials_smoothed = \
    np.swapaxes(np.array(sr_trials_smoothed), 0, 1)
sr_baseline_trials_smoothed = \
    np.swapaxes(np.array(sr_baseline_trials_smoothed), 0, 1)
sr_non_trials_smoothed = \
    np.swapaxes(np.array(sr_non_trials_smoothed), 0, 1)
sr_baseline_non_trials_smoothed = \
    np.swapaxes(np.array(sr_baseline_non_trials_smoothed), 0, 1)


# Average the (smoothed) spike rates over the events
#   Averages and stds the per frame spike rates
sr_average_around_trials = sr_trials.mean(1)
sr_average_around_trials_std = sr_trials.std(1)

sr_average_around_non_trials = sr_non_trials.mean(1)
sr_average_around_non_trials_std = sr_non_trials.std(1)

sr_average_around_baseline_trials = sr_baseline_trials.mean(1)
sr_average_around_baseline_trials_std = sr_baseline_trials.std(1)

sr_average_around_baseline_non_trials = sr_baseline_non_trials.mean(1)
sr_average_around_baseline_non_trials_std = sr_baseline_non_trials.std(1)

#   Average and std the smoothed spike rates
sr_average_around_trials_smoothed = sr_trials_smoothed.mean(1)
sr_average_around_trials_smoothed_var = sr_trials_smoothed.var(1)

sr_average_around_non_trials_smoothed = sr_non_trials_smoothed.mean(1)
sr_average_around_non_trials_smoothed_var = sr_non_trials_smoothed.var(1)

sr_average_around_baseline_trials_smoothed = sr_baseline_trials_smoothed.mean(1)
sr_average_around_baseline_trials_smoothed_var = sr_baseline_trials_smoothed.var(1)

sr_average_around_baseline_non_trials_smoothed = sr_baseline_non_trials_smoothed.mean(1)
sr_average_around_baseline_non_trials_smoothed_var = sr_baseline_non_trials_smoothed.var(1)


#   Create the (non-equal variance) t-test for each time bin and neuron for each baseline vs non-baseline pair
sigma_triangle_trials = np.sqrt((sr_average_around_trials_smoothed_var /
                                 number_of_trial_pokes) +
                                (sr_average_around_baseline_trials_smoothed_var /
                                 number_of_trial_pokes))

t_test_trials = (sr_average_around_trials_smoothed - sr_average_around_baseline_trials_smoothed)\
                / sigma_triangle_trials

neurons_that_change_over_trials = np.unique(np.argwhere(np.logical_or(t_test_trials > 3, t_test_trials < -3))[:, 0])

sigma_triangle_non_trials = np.sqrt((sr_average_around_non_trials_smoothed_var /
                                     number_of_non_trial_pokes) +
                                    (sr_average_around_baseline_non_trials_smoothed_var /
                                     number_of_non_trial_pokes))

t_test_non_trials = (sr_average_around_non_trials_smoothed - sr_average_around_baseline_non_trials_smoothed)\
                / sigma_triangle_non_trials

neurons_that_change_over_non_trials = \
    np.unique(np.argwhere(np.logical_or(t_test_non_trials > 3, t_test_non_trials < -3))[:, 0])

#   PCA
#       PCA the smoothed average spike rates
pca_trials_smoothed = PCA()
_ = pca_trials_smoothed.fit_transform(sr_average_around_trials_smoothed[neurons_that_change_over_trials, :])
pca_all_neurons_avg_around_trials_smoothed = pca_trials_smoothed.components_

pca_non_trials_smoothed = PCA()
_ = pca_non_trials_smoothed.fit_transform(sr_average_around_non_trials_smoothed[neurons_that_change_over_non_trials, :])
pca_all_neurons_avg_around_non_trials_smoothed = pca_non_trials_smoothed.components_

#       PCA the t-test scores of the smoothed averaged spike rates
pca_trials_smoothed = PCA()
t_test_trials[np.isnan(t_test_trials)] = 0
_ = pca_trials_smoothed.fit_transform(t_test_trials[neurons_that_change_over_trials, :])
pca_t_test_trials_smoothed = pca_trials_smoothed.components_

pca_non_trials_smoothed = PCA()
t_test_non_trials[np.isnan(t_test_non_trials)] = 0
_ = pca_non_trials_smoothed.fit_transform(t_test_non_trials[neurons_that_change_over_non_trials, :])
pca_t_test_non_trials_smoothed = pca_non_trials_smoothed.components_


dataset1 = pca_all_neurons_avg_around_trials_smoothed
dataset2 = pca_all_neurons_avg_around_non_trials_smoothed

dataset1 = pca_t_test_trials_smoothed
dataset2 = pca_t_test_non_trials_smoothed


cm1 = cm.binary
color1 = [cm1(c) for c in np.arange(0, 0.5, 1 / (2 * len(pca_all_neurons_avg_around_trials_smoothed[0])))]
color1[int(len(pca_all_neurons_avg_around_trials_smoothed[0]) / 2)] = (0, 0, 0)
color2 = [cm1(c) for c in np.arange(1, 0.5, -1 / (2 * len(pca_all_neurons_avg_around_trials_smoothed[0])))]
color2[int(len(pca_all_neurons_avg_around_trials_smoothed[0]) / 2)] = (0, 0, 0)
color = [cm1(c) for c in np.arange(0, 1, 1 / (len(pca_all_neurons_avg_around_trials_smoothed[0])))]
color[int(len(pca_all_neurons_avg_around_trials_smoothed[0]) / 2)] = (0, 0, 0)

i = 0
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=dataset1[i, :], ys=dataset1[i + 1, :], zs=dataset1[i + 2, :], c=color)
ax.plot(xs=dataset1[i, :], ys=dataset1[i + 1, :], zs=dataset1[i + 2, :])
ax.scatter(xs=dataset2[i, :], ys=dataset2[i + 1, :], zs=dataset2[i + 2, :], c=color)
ax.plot(xs=dataset2[i, :], ys=dataset2[i + 1, :], zs=dataset2[i + 2, :])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset1[i, :], dataset1[i + 1, :], c=color)
ax.plot(dataset1[i, :], dataset1[i + 1, :])
ax.scatter(dataset2[i, :], dataset2[i + 1, :], c=color)
ax.plot(dataset2[i, :], dataset2[i + 1, :])

distances_in_pca_space = np.zeros(dataset1.shape[1])
for d in np.arange(dataset1.shape[1]):
    distances_in_pca_space[d] = \
        np.sqrt((dataset1[0, d] - dataset2[0, d])**2 + (dataset1[1, d] - dataset2[1, d])**2 )

plt.plot(np.arange(-len(distances_in_pca_space)*step_window/240, len(distances_in_pca_space)*step_window/240,
                   step_window*0.00833)[:-1],
         distances_in_pca_space)
plt.vlines(x=0, ymin=distances_in_pca_space.min(), ymax=distances_in_pca_space.max())

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="DIM RED SPIKE RATES WITH RANDOM PROJECTION AND TSNE THAT">

window_size_seconds = 5
window_size_time_points = window_size_seconds * const.SAMPLING_FREQUENCY

bin_size_in_seconds = 0.004
bin_size_in_time_points = bin_size_in_seconds * const.SAMPLING_FREQUENCY

time_points_of_all_pokes = np.concatenate((time_points_of_trial_pokes, time_points_of_non_trial_pokes))

# Get the unique combinations of cells firing with a time bin of 4 ms
number_of_neurons = len(all_neuron_templates)
number_of_pokes = len(time_points_of_all_pokes)
number_of_bins = int(2 * window_size_time_points / bin_size_in_time_points)
neurons_spiking_4ms_bins = np.zeros((number_of_pokes,
                                     number_of_neurons,
                                     number_of_bins), dtype=np.bool)

for n in np.arange(len(all_neuron_templates)):
    neuron_template = all_neuron_templates[n]
    neuron_spikes_times = spike_info[spike_info['template_after_sorting'] == neuron_template]['times'].values
    for sp in np.arange(len(time_points_of_all_pokes)):
        start_point = time_points_of_all_pokes[sp] - window_size_time_points
        shifted_spikes = neuron_spikes_times.astype(np.int32) - start_point
        first_spike_index = np.argwhere(shifted_spikes > 0)
        if len(first_spike_index) > 0:
            first_spike_index = first_spike_index[0][0]
            last_spike_index = np.argwhere(shifted_spikes < 2 * window_size_time_points)
            if len(last_spike_index) > 0:
                last_spike_index = last_spike_index[-1][0]
                temp = shifted_spikes[first_spike_index:last_spike_index + 1]
                neurons_spiking_4ms_bins[sp, n, np.floor(temp/bin_size_in_time_points).astype(np.int)] = 1

# Have a look
fig = plt.figure()
ax = fig.add_subplot(111)


def show_trial(t):
    ax.clear()
    ax.imshow(neurons_spiking_4ms_bins[t, :, :], aspect='auto')
    return None


trial = 0
out = None
sl.connect_repl_var(globals(), 'trial', 'out', 'show_trial', slider_limits=[0, len(time_points_of_all_pokes) - 1])


def bin2int(x):
    y = np.int64(0)
    t = x.astype(np.int64)
    for i, j in enumerate(t):
        y += j << i
    return y


codes_of_firing = np.zeros((number_of_pokes, number_of_bins), dtype=np.int64)
for t in np.arange(number_of_pokes):
    codes_of_firing[t, :] = ([bin2int(neurons_spiking_4ms_bins[t, :, b]) for b in np.arange(number_of_bins)])

unique_codes_of_firing = np.unique(codes_of_firing)

codes_of_firing_before_poke = codes_of_firing[:, :int(number_of_bins/2)]
codes_of_firing_after_poke = codes_of_firing[:, int(number_of_bins/2):]

unique_codes_of_firing_before_poke = np.unique(codes_of_firing_before_poke)
unique_codes_of_firing_after_poke = np.unique(codes_of_firing_after_poke)

common_codes_of_firing = unique_codes_of_firing_before_poke[np.in1d(unique_codes_of_firing_before_poke,
                                                                    unique_codes_of_firing_after_poke)]

print('Unique firing combinations found in the whole time span = {}'.format(len(unique_codes_of_firing)))
print('Unique firing combinations found before the poke = {}'.format(len(unique_codes_of_firing_before_poke)))
print('Unique firing combinations found after the poke = {}'.format(len(unique_codes_of_firing_after_poke)))
print('Common firing combinations between before and after the poke = {}'.format(len(common_codes_of_firing)))

codes_of_firing_with_unique_base = np.empty((number_of_pokes, number_of_bins))
for p in np.arange(number_of_pokes):
    for b in np.arange(number_of_bins):
        codes_of_firing_with_unique_base[p, b] = \
            np.argwhere(np.in1d(unique_codes_of_firing, codes_of_firing[p, b])).item()


c = ['k']*number_of_pokes
c[:42] = ['r']*42
codes_of_firing_with_unique_base_embedded = TSNE(n_components=2).fit_transform(codes_of_firing_with_unique_base)
plt.scatter(codes_of_firing_with_unique_base_embedded[:, 0], codes_of_firing_with_unique_base_embedded[:, 1], c=c)

random_projection_transformer = random_projection.SparseRandomProjection(eps=0.3)
firing_codes_random_projection = random_projection_transformer.fit_transform(codes_of_firing_with_unique_base)

c = ['k']*number_of_pokes
c[:42] = ['r']*42
firing_codes_random_projection_embedded = TSNE(n_components=2, n_iter=4000, angle=0.1).fit_transform(firing_codes_random_projection)
plt.scatter(codes_of_firing_with_unique_base_embedded[:, 0], firing_codes_random_projection_embedded[:, 1], c=c)


codes_of_firing_with_unique_base_before_poke = codes_of_firing_with_unique_base[:, :int(number_of_bins/2)]
codes_of_firing_with_unique_base_after_poke = codes_of_firing_with_unique_base[:, int(number_of_bins/2):]

random_projection_transformer1 = random_projection.SparseRandomProjection(eps=0.26)
firing_codes_before_poke_random_projection = \
    random_projection_transformer1.fit_transform(codes_of_firing_with_unique_base_before_poke)
firing_codes_before_poke_random_projection_embedded = \
    TSNE(n_components=2, n_iter=20000, angle=0.1).fit_transform(firing_codes_before_poke_random_projection)
plt.scatter(firing_codes_before_poke_random_projection_embedded[:, 0],
            firing_codes_before_poke_random_projection_embedded[:, 1], c=c)

random_projection_transformer2 = random_projection.SparseRandomProjection(eps=0.3)
firing_codes_after_poke_random_projection = \
    random_projection_transformer2.fit_transform(codes_of_firing_with_unique_base_after_poke)
firing_codes_after_poke_random_projection_embedded = \
    TSNE(n_components=2, n_iter=20000, angle=0.1, perplexity=50).fit_transform(firing_codes_after_poke_random_projection)
c = ['b']*number_of_pokes
c[:42] = ['y']*42
plt.scatter(firing_codes_after_poke_random_projection_embedded[:, 0],
            firing_codes_after_poke_random_projection_embedded[:, 1], c=c)


pca_before = PCA()
_ = pca_before.fit_transform(codes_of_firing_with_unique_base_before_poke)
firing_codes_before_poke_pca = pca_before.components_
firing_codes_before_poke_pca_embedded = \
    TSNE(n_components=2, n_iter=20000, angle=0.1).fit_transform(firing_codes_before_poke_pca)
c = ['k']*number_of_pokes
c[:42] = ['r']*42
plt.scatter(firing_codes_before_poke_pca_embedded[:, 0],
            firing_codes_before_poke_pca_embedded[:, 1], c=c)

pca_after = PCA()
_ = pca_after.fit_transform(codes_of_firing_with_unique_base_after_poke)
firing_codes_after_poke_pca = pca_after.components_
firing_codes_after_poke_pca_embedded = \
    TSNE(n_components=2, n_iter=10000, angle=0.1, perplexity=20).fit_transform(firing_codes_after_poke_pca)
c = ['b']*number_of_pokes
c[:42] = ['y']*42
plt.scatter(firing_codes_after_poke_pca_embedded[:, 0],
            firing_codes_after_poke_pca_embedded[:, 1], c=c)

# </editor-fold>


from scipy import ndimage

sums = np.zeros((number_of_pokes, number_of_pokes))
for t in np.arange(number_of_pokes):
    for i in np.arange(number_of_pokes):
        temp = ndimage.convolve(neurons_spiking_4ms_bins[t, :, :], neurons_spiking_4ms_bins[i, :, :])
        sums[t, i] = temp.sum()
        print(t, i)


np.save(join(poke_folder, 'sums_of_concolutions_of_firing_patterns.npy'), sums)
