

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis.Statistics import binning, cluster_based_permutation_tests as cl_per
from BrainDataAnalysis.Graphics import ploting_functions as plf

from sklearn import preprocessing as preproc
import pandas as pd
import matplotlib.pyplot as plt

import common_data_transforms as cdt
import sequence_viewer as sv
import slider as sl

from npeet.lnc import MI


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)

raw_data_filename = join(data_folder, 'Amplifier_APs.bin')
raw_data = ns_funcs.load_binary_amplifier_data(raw_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

# </editor-fold>
# -------------------------------------------------
# <editor-fold desc="GET TIMES AND FRAMES OF SUCCESSFUL TRIALS">

camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
sounds_dur = sounds[:, 1] - sounds[:, 0]
reward_sounds = sounds[sounds_dur < 4000]

# Using the trialend csv file to generate events
# succesful_trials = event_dataframes['ev_trial_end'][event_dataframes['ev_trial_end']['Result'] == 'Food']
# succesful_trials =  succesful_trials['AmpTimePoints'].values

# Using the start of the reward tone to generate events
# There is a difference of 78.6 frames (+-2) between the reward tone and the csv file event (about 700ms)
succesful_trials = reward_sounds[:, 0]

# Get the average firing rates of all neurons a few seconds around the successful pokes
time_around_beam_break = 8
avg_firing_rate_around_suc_trials = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=succesful_trials,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break)
# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="NEURONS THAT INCREASE THEIR FIRING RATES AROUND THE POKE EVENT">
# Find which neurons increase their firing rate on average around a successful poke
increasing_firing_rates_neuron_index, increasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_suc_trials,
                                                           time_around_pattern=time_around_beam_break,
                                                           pattern_regions_to_compare=[0, 0.8, 1.0, 1.2],
                                                           comparison_factor=4, comparison_direction='increase',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(increasing_firing_rates)

# Show where the neurons are in the brain
template_info_increasing_fr_neurons = template_info.iloc[increasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_increasing_fr_neurons)

pd.to_pickle(template_info_increasing_fr_neurons, join(poke_folder, 'ti_increasing_neurons_on_trial_pokes.df'))

# Have a detailed look at the neuron with the largest increase
# largest_increase_neuron_index = increasing_firing_rates_neuron_index[np.argmax(increasing_firing_rates_ratio)]
time_around_beam_break = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
frames_around_beam_break = 120 * time_around_beam_break
time_points_around_beam_break = const.SAMPLING_FREQUENCY * time_around_beam_break
args = [increasing_firing_rates_neuron_index, avg_firing_rate_around_suc_trials, template_info, spike_info,
        succesful_trials, frames_around_beam_break, fig1, fig2]

show_rasters_increase = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_increase', 'args',
                    slider_limits=[0, len(increasing_firing_rates_neuron_index) - 1])

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="NEURONS THAT DECREASE THEIR FIRING RATES AROUND THE POKE EVENT">
# Find which neurons decrease their firing rate on average around a successful poke
decreasing_firing_rates_neuron_index, decreasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rate_around_suc_trials,
                                                           time_around_pattern=time_around_beam_break,
                                                           pattern_regions_to_compare=[0, 0.8, 1.0, 1.2],
                                                           comparison_factor=4, comparison_direction='decrease',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(decreasing_firing_rates)

# Show where the neurons are in the brain
template_info_decreasing_fr_neurons = template_info.iloc[decreasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_decreasing_fr_neurons)

pd.to_pickle(template_info_decreasing_fr_neurons, join(poke_folder, 'ti_decreasing_neurons_on_trial_pokes.df'))

# Have a look at the neuron with the largest decrease
time_around_beam_break = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
frames_around_beam_break = 120 *time_around_beam_break
args = [decreasing_firing_rates_neuron_index, avg_firing_rate_around_suc_trials, template_info, spike_info,
        succesful_trials, frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(decreasing_firing_rates_neuron_index) - 1])

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Looking at the time of touching ball. Nothing happens!">
plt.plot(cdt.space_data(avg_firing_rate_around_suc_trials, 10).transpose())

touch_ball_trials = event_dataframes['ev_rat_touch_ball']

frames_of_touch_ball_trials = sync_funcs.time_point_to_frame_from_video_df(event_dataframes['ev_video'],
                                                                           touch_ball_trials['AmpTimePoints'].values)

firing_rate_around_touch_trials = np.zeros((len(touch_ball_trials), len(spike_rates), 2 * frames_around_beam_break))

for f in np.arange(len(firing_rate_around_touch_trials)):
    frame = frames_of_touch_ball_trials[f]
    firing_rate_around_touch_trials[f, :, :] = spike_rates[:, frame - frames_around_beam_break:
                                                            frame + frames_around_beam_break]

avg_firing_rate_around_touch_trials = firing_rate_around_touch_trials.mean(axis=0)
avg_firing_rate_around_touch_trials -= np.expand_dims(avg_firing_rate_around_touch_trials[:, :600].mean(axis=1), axis=1)
plt.imshow(avg_firing_rate_around_touch_trials, vmax=avg_firing_rate_around_touch_trials.max(), vmin=0)

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Generate raster plots for all neurons around poke event">

all_pic_folder = join(results_folder, 'Images', 'Modulating_Neurons_On_Events', 'All_neurons')
fig = plt.figure(0, figsize=(8, 6), dpi=150)
time_around_beam_break = 8
frames_around_beam_break = 120 * time_around_beam_break

dec_and_inc_neurons_indices = np.union1d(decreasing_firing_rates_neuron_index, increasing_firing_rates_neuron_index)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)
for index in position_sorted_indices:
    fig = fr_funcs.show_rasters(index, template_info, spike_info,
                                succesful_trials, frames_around_beam_break, fig)
    template_number = template_info.iloc[index]['template number']
    firing_rate = "%3.3f" % template_info.iloc[index]['firing rate']
    y_position = int(template_info.iloc[index]['position Y'] * const.POSITION_MULT)
    in_group = 0
    if index in dec_and_inc_neurons_indices:
        in_group = 1
    plt.title('Template number = {}, Height on probe = {}, Firing rate = {} Hz, In group = {}'.format(str(template_number),
                                                                                                 str(y_position),
                                                                                                 firing_rate,
                                                                                                 str(in_group)))
    plt.savefig(join(all_pic_folder, 'height_{}_template_{}_index_{}.png'.format(str(y_position), str(template_number),
                                                                                 str(index))))

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Create the average of the voltage of all high pass electrodes around the trial pokes">

time_around_beam_break = 0.1
timepoints_around_beam_break = int(const.SAMPLING_FREQUENCY * time_around_beam_break)

voltage_around_trials = np.empty((len(succesful_trials), const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                                  2 * timepoints_around_beam_break))
for st in np.arange(len(succesful_trials)):
    voltage_around_trials[st, :, :] = raw_data[:, succesful_trials[st] - timepoints_around_beam_break:
                                                  succesful_trials[st] + timepoints_around_beam_break]

avg_voltage_around_trials = np.mean(voltage_around_trials, axis=0)

_ = plt.imshow(np.flipud(avg_voltage_around_trials), aspect='auto')
# </editor-fold>


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


plt.imshow(np.flipud(tns), aspect='auto', extent=[-8, 8, len(tns), 0])
plt.hlines(y=len(t) - region_lines, xmin=-8, xmax=8, linewidth=2, color='w')
plt.vlines(x=0, ymin=0, ymax=len(tns) - 1)


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
        succesful_trials, frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(avg_firing_rate_around_suc_trials) - 1])
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="CHECK FOR CORRELATIONS BETWEEN SPIKES RATES OF ALL NEURONS AROUND EVENT">

fr_norm_sorted = tns

cov = np.cov(fr_norm_sorted)


# MUTUAL INFO BETWEEN RATES
n = 0
mutual_infos = []
for rate1 in fr_norm_sorted:
    for rate2 in fr_norm_sorted:
        mutual_infos.append(MI.mi_LNC([rate1.tolist(), rate2.tolist()],
                                            k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))


mutual_infos = np.reshape(np.array(mutual_infos), (fr_norm_sorted.shape[0], fr_norm_sorted.shape[0]))
np.save(join(analysis_folder, 'Results', 'MutualInformation',
             'mutual_infos_NSprobe_spike_rates_vs_spike_rates_all_neurons_around_succ_trials.npy'), mutual_infos)

# </editor-fold>


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc="COMPARE THE FRs AROUND THE POKE EVENTS WITH THE ONES AROUND THE RANDOM ONES">

time_around_beam_break = 8
firing_rate_around_suc_trials, _ = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=succesful_trials,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break,
                                                                                keep_trials=True)

events_random = np.random.choice(np.arange(succesful_trials.min(), succesful_trials.max(), 100),
                                 len(succesful_trials), replace=False)
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

plt.figure(0)
plt.imshow(np.flipud(np.mean(timebined_sorted_special_normalised_frs_around_suc_trials, axis=0)), aspect='auto')
plt.hlines(y=neurons - region_lines, xmin=0, xmax=timebins-1, linewidth=3, color='w')
plt.vlines(x=int(timebins / 2), ymin=0, ymax=neurons - 1)

timebined_sorted_frs_around_suc_trials_tr = np.transpose(timebined_sorted_frs_around_suc_trials, [1, 2, 0])
timebined_sorted_frs_around_random_tr = np.transpose(timebined_sorted_frs_around_random, [1, 2, 0])
p_values_pokes_vs_random, cluster_labels_poke_vs_random = \
    cl_per.monte_carlo_significance_probability(timebined_sorted_frs_around_suc_trials_tr, timebined_sorted_frs_around_random_tr,
                                                num_permutations=1000, min_area=8, cluster_alpha=0.05,
                                                monte_carlo_alpha=0.05, sample_statistic='independent',
                                                cluster_statistic='maxsum')

data = avg_timebined_sorted_normalised_frs_around_suc_trials
cluster_labels = cluster_labels_poke_vs_random
plf.show_significant_clusters_on_data(data, cluster_labels, region_lines, np.arange(neurons), window_time=8,
                                      colormap='binary', markers='o', alpha=0.6)


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

