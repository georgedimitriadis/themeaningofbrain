


from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis.Statistics import binning, cluster_based_permutation_tests as cl_per
from BrainDataAnalysis.Graphics import ploting_functions as plf

from sklearn import preprocessing as preproc
import pandas as pd
import matplotlib.pyplot as plt

import sequence_viewer as sv
import slider as sl



# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 5
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Denoised', 'Kilosort')

events_folder = join(data_folder, "events")

results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')

# event_dataframes = ns_funcs.load_events_dataframes(events_folder, [sync_funcs.event_types[-1]])
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

# spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cortex_sorting.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="GET TIMES AND FRAMES OF POKE TOUCHES WITHOUT TOUCHING THE BALL FIRST">

camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

time_to_next_poke = np.diff(beam_breaks[:, 0])/const.SAMPLING_FREQUENCY
minimum_delay = 3
pokes_after_delay = np.squeeze(beam_breaks[np.argwhere(time_to_next_poke > minimum_delay)])
start_pokes_after_delay = pokes_after_delay[:, 0]

# Check if any of the pokes after delay are also in the spaces between touching the ball and the reward sound playing
# and remove those
sounds_dur = sounds[:, 1] - sounds[:, 0]
reward_sounds = sounds[sounds_dur < 4000]
start_reward_sounds = reward_sounds[:, 0]

overlaps = []
for r_sound in start_reward_sounds:
    t = np.logical_and(start_pokes_after_delay > r_sound - 4*const.SAMPLING_FREQUENCY,
                       start_pokes_after_delay < r_sound + 4*const.SAMPLING_FREQUENCY)
    if np.any(t):
        overlaps.append(np.squeeze(np.argwhere(t)))

overlaps = np.squeeze(np.array(overlaps))
print(overlaps)
print(len(overlaps))

start_pokes_after_delay = np.delete(start_pokes_after_delay, overlaps)
# start_pokes_after_delay = start_pokes_after_delay[2:]

np.save(join(poke_folder, 'time_points_of_non_trial_pokes.npy'), start_pokes_after_delay )

time_around_beam_break = 8
avg_firing_rate_around_suc_trials = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=start_pokes_after_delay,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break)

increasing_firing_rates_neuron_index, increasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_suc_trials,
                                                           time_around_pattern=time_around_beam_break,
                                                           pattern_regions_to_compare=[0, 0.6, 0.8, 1.2],
                                                           comparison_factor=3, comparison_direction='increase',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(increasing_firing_rates)

# Show where the neurons are in the brain
template_info_increasing_fr_neurons = template_info.iloc[increasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_increasing_fr_neurons)

pd.to_pickle(template_info_increasing_fr_neurons, join(poke_folder, 'ti_increasing_neurons_on_non_trial_pokes.df'))


decreasing_firing_rates_neuron_index, decreasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_suc_trials,
                                                           time_around_pattern=time_around_beam_break,
                                                           pattern_regions_to_compare=[0, 0.6, 0.8, 1.2],
                                                           comparison_factor=3, comparison_direction='decrease',
                                                           baserate=0.5)

fr_funcs.show_firing_rates_around_event(decreasing_firing_rates)


# Show where the neurons are in the brain
template_info_decreasing_fr_neurons = template_info.iloc[decreasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_decreasing_fr_neurons)

pd.to_pickle(template_info_decreasing_fr_neurons, join(poke_folder, 'ti_decreasing_neurons_on_non_trial_pokes.df'))



# -------------------------------------------------
# <editor-fold desc="LOOK AT ALL THE NEURONS AROUND THE POKE EVENT">

smooth_time = 0.5
smooth_frames = smooth_time * 120

t = binning.rolling_window_with_step(avg_firing_rate_around_suc_trials, np.mean, smooth_frames, int(smooth_frames / 3))
tn = preproc.normalize(t, norm='l1', axis=0)

tn = np.asarray(t)
for i in np.arange(len(t)):
    tn[i, :] = binning.scale(t[i], 0, 1)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)

regions_pos = list(const.BRAIN_REGIONS.values())
region_lines = []
for rp in regions_pos:
    region_lines.append(sync_funcs.find_nearest(y_positions[position_sorted_indices] * const.POSITION_MULT, rp)[0])
region_lines = np.array(region_lines)

tns = tn[position_sorted_indices]

plt.imshow(np.flipud(tns), aspect='auto')
plt.hlines(y=len(t) - region_lines, xmin=0, xmax=len(tns[0])-1, linewidth=3, color='w')
plt.vlines(x=int(len(tns[0]) / 2), ymin=0, ymax=len(tns) - 1)


i = 0
sv.graph_pane(globals(), 'i', 'tn')


time_around_beam_break = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
all_indices = np.arange(len(avg_firing_rate_around_suc_trials))
frames_around_beam_break = 120 *time_around_beam_break
args = [all_indices, avg_firing_rate_around_suc_trials, template_info, spike_info,
        start_pokes_after_delay, frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(avg_firing_rate_around_suc_trials) - 1])
# </editor-fold>



# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc="COMPARE THE FRs AROUND THE POKE EVENTS WITH THE ONES AROUND THE RANDOM ONES">

minimum_delay = 5
start_pokes_after_delay = np.load(join(events_definitions_folder,
                                       'events_first_pokes_after_{}_delay_non_reward.npy'.format(str(minimum_delay))))
start_pokes_after_delay = start_pokes_after_delay[:-1]
time_around_beam_break = 8
#non_succ_trials = np.random.choice(start_pokes_after_delay, 42, replace=False)
non_succ_trials = start_pokes_after_delay
firing_rate_around_suc_trials, _ = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=non_succ_trials,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break,
                                                                                keep_trials=True)

events_random = np.random.choice(np.arange(non_succ_trials.min(), non_succ_trials.max(), 100),
                                 len(non_succ_trials), replace=False)
firing_rate_around_random_times, _ = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=events_random,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break,
                                                                                 keep_trials=True)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)

regions_pos = list(const.BRAIN_REGIONS.values())
region_lines = []
for rp in regions_pos:
    region_lines.append(sync_funcs.find_nearest(y_positions[position_sorted_indices] * const.POSITION_MULT, rp)[0])
region_lines = np.array(region_lines)

smooth_time = 0.5
smooth_frames = smooth_time * 120

t = binning.rolling_window_with_step(firing_rate_around_suc_trials[0, :, :], np.mean, smooth_frames,
                                         int(smooth_frames / 3))


trials = firing_rate_around_suc_trials.shape[0]
neurons = firing_rate_around_suc_trials.shape[1]
timebins = t.shape[1]
timebined = np.empty((trials, neurons, timebins))
timebined_sorted_frs_around_suc_trials = np.empty((trials, neurons, timebins))
timebined_sorted_frs_around_random = np.empty((trials, neurons, timebins))
timebined_sorted_normalised_frs_around_suc_trials = np.empty((trials, neurons, timebins))
timebined_sorted_normalised_frs_around_random = np.empty((trials, neurons, timebins))

for trial in np.arange(trials):
    timebined[trial, :, :] = binning.rolling_window_with_step(firing_rate_around_suc_trials[trial, :, :], np.mean,
                                                              smooth_frames, int(smooth_frames / 3))

    timebined_sorted = timebined[trial, position_sorted_indices, :]

    timebined_sorted_frs_around_suc_trials[trial, :, :] = timebined_sorted

    for neuron in np.arange(neurons):
        timebined_sorted_normalised_frs_around_suc_trials[trial, neuron, :] = \
            binning.scale(timebined_sorted_frs_around_suc_trials[trial, neuron, :], 0, 1)

    timebined[trial, :, :] = binning.rolling_window_with_step(firing_rate_around_random_times[trial, :, :], np.mean,
                                                              smooth_frames, int(smooth_frames / 3))

    timebined_sorted = timebined[trial, position_sorted_indices, :]

    timebined_sorted_frs_around_random[trial, :, :] = timebined_sorted

    for neuron in np.arange(neurons):
        timebined_sorted_normalised_frs_around_random[trial, neuron, :] = \
            binning.scale(timebined_sorted_frs_around_random[trial, neuron, :], 0, 1)

avg_timebined_sorted_frs_around_suc_trials = np.mean(timebined_sorted_frs_around_suc_trials, axis=0)
avg_timebined_sorted_frs_around_random = np.mean(timebined_sorted_frs_around_random, axis=0)

avg_timebined_sorted_normalised_frs_around_suc_trials = np.empty((neurons, timebins))
for neuron in np.arange(neurons):
    avg_timebined_sorted_normalised_frs_around_suc_trials[neuron, :] = \
        binning.scale(avg_timebined_sorted_frs_around_suc_trials[neuron, :], 0, 1)

avg_timebined_sorted_normalised_frs_around_random = np.empty((neurons, timebins))
for neuron in np.arange(neurons):
    avg_timebined_sorted_normalised_frs_around_random[neuron, :] = \
        binning.scale(avg_timebined_sorted_frs_around_random[neuron, :], 0, 1)


# Normalising the average is not the same as averaging the normalised trials!
# The following scales individual trials so when they get averaged the result is going to be normalised
timebined_sorted_special_normalised_frs_around_suc_trials = np.empty((trials, neurons, timebins))
timebined_sorted_special_normalised_frs_around_random = np.empty((trials, neurons, timebins))

for neuron in np.arange(neurons):

    X_max_suc = np.mean(timebined_sorted_frs_around_suc_trials, axis=0)[neuron].max()
    X_min_suc = np.mean(timebined_sorted_frs_around_suc_trials, axis=0)[neuron].min()

    X_max_rand = np.mean(timebined_sorted_frs_around_random, axis=0)[neuron].max()
    X_min_rand = np.mean(timebined_sorted_frs_around_random, axis=0)[neuron].min()

    for trial in np.arange(trials):

        timebined_sorted_special_normalised_frs_around_suc_trials[trial, neuron] = \
            binning.scale(timebined_sorted_frs_around_suc_trials[trial, neuron, :], 0, 1, X_min_suc, X_max_suc)

        timebined_sorted_special_normalised_frs_around_random[trial, neuron] = \
            binning.scale(timebined_sorted_frs_around_random[trial, neuron, :], 0, 1, X_min_rand, X_max_rand)


plt.figure(4)
plt.imshow(np.flipud(np.mean(timebined_sorted_special_normalised_frs_around_suc_trials, axis=0)), aspect='auto')
plt.hlines(y=neurons - region_lines, xmin=0, xmax=timebins-1, linewidth=3, color='w')
plt.vlines(x=int(timebins / 2), ymin=0, ymax=neurons - 1)
plt.colorbar()

plt.figure(5)
plt.imshow(np.flipud(np.mean(timebined_sorted_special_normalised_frs_around_random, axis=0)), aspect='auto')
plt.hlines(y=neurons - region_lines, xmin=0, xmax=timebins-1, linewidth=3, color='w')
plt.vlines(x=int(timebins / 2), ymin=0, ymax=neurons - 1)


timebined_sorted_frs_around_suc_trials_tr = np.transpose(timebined_sorted_normalised_frs_around_suc_trials, [1, 2, 0])
timebined_sorted_frs_around_random_tr = np.transpose(timebined_sorted_normalised_frs_around_random, [1, 2, 0])
p_values_pokes_vs_random, cluster_labels_poke_vs_random = \
    cl_per.monte_carlo_significance_probability(timebined_sorted_frs_around_suc_trials_tr, timebined_sorted_frs_around_random_tr,
                                                num_permutations=5000, min_area=4, cluster_alpha=0.05,
                                                monte_carlo_alpha=0.05, sample_statistic='independent',
                                                cluster_statistic='maxarea')

data = avg_timebined_sorted_normalised_frs_around_suc_trials
cluster_labels = cluster_labels_poke_vs_random
plf.show_significant_clusters_on_data(data, cluster_labels, region_lines, np.arange(neurons), window_time=8,
                                      colormap='binary', markers='o', alpha=0.6, marker_color='b')
plt.vlines(x=0, ymin=0, ymax=neurons - 1)
plt.title(const.rat_folder)

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


