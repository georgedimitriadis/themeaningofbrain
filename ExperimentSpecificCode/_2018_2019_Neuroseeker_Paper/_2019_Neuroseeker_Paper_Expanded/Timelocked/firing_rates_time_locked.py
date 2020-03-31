

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._47p2 \
    import constants_47p2 as const
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded \
    import constants_common as const_comm
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis.Statistics import binning

from sklearn import preprocessing as preproc
import pandas as pd
import matplotlib.pyplot as plt

import common_data_transforms as cdt
import sequence_viewer as sv
import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)

raw_data_filename = join(data_folder, 'Amplifier_APs.bin')
raw_data = ns_funcs.load_binary_amplifier_data(raw_data_filename,
                                               number_of_channels=const_comm.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

# </editor-fold>
# -------------------------------------------------
# <editor-fold desc="GET TIMES AND FRAMES AROUND DIFFERENT TRIALS">

trials = {'s': np.load(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')),
          'tb': np.load(join(events_definitions_folder, 'events_touch_ball.npy'))}

if date_folder != 6:
    minimum_delay = 5
    nst = np.load(join(events_definitions_folder,
                                       'events_first_pokes_after_{}_delay_non_reward.npy'.format(str(minimum_delay))))
    if date_folder == 8:
        nst = nst[1:]

    trials['ns'] = nst

trials['r'] = np.random.choice(np.arange(200000, raw_data.shape[1]-200000), len(trials['s']))

# Get the average firing rates of all neurons a few seconds around the trials
time_before_and_after_event = 8

avg_firing_rates = {}
for t in ['s', 'tb', 'r']:
    avg_firing_rates[t] = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                      event_time_points=trials[t],
                                                                      ev_video_df=event_dataframes['ev_video'],
                                                                      time_around_event=time_before_and_after_event)

if date_folder != 6:
    avg_firing_rates['ns'] = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                         event_time_points=trials['ns'],
                                                                         ev_video_df=event_dataframes['ev_video'],
                                                                         time_around_event=time_before_and_after_event)


# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="NEURONS THAT INCREASE THEIR FIRING RATES AROUND THE POKE EVENT">
# Find which neurons increase their firing rate on average around a successful poke

trial_type = 'r'  # Possible 's', 'ns', 'tb'

increasing_firing_rates_neuron_index, increasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rates[trial_type],
                                                           time_around_pattern=time_before_and_after_event,
                                                           pattern_regions_to_compare=[0, 0.8, 1.0, 1.2],
                                                           comparison_factor=4, comparison_direction='increase',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(increasing_firing_rates)

# Show where the neurons are in the brain
template_info_increasing_fr_neurons = template_info.iloc[increasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const_comm.PROBE_DIMENSIONS,
                                     const_comm.POSITION_MULT, template_info=template_info_increasing_fr_neurons)

pd.to_pickle(template_info_increasing_fr_neurons, join(poke_folder, 'ti_increasing_neurons_on_trial_pokes.df'))

# Have a detailed look at the neuron with the largest increase
# largest_increase_neuron_index = increasing_firing_rates_neuron_index[np.argmax(increasing_firing_rates_ratio)]
time_before_and_after_event = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
frames_around_beam_break = 120 * time_before_and_after_event
time_points_around_beam_break = const_comm.SAMPLING_FREQUENCY * time_before_and_after_event
args = [increasing_firing_rates_neuron_index, avg_firing_rates[trial_type], template_info, spike_info,
        trials[trial_type], frames_around_beam_break, fig1, fig2]

show_rasters_increase = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_increase', 'args',
                    slider_limits=[0, len(increasing_firing_rates_neuron_index) - 1])

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="NEURONS THAT DECREASE THEIR FIRING RATES AROUND THE POKE EVENT">
# Find which neurons decrease their firing rate on average around a successful poke

decreasing_firing_rates_neuron_index, decreasing_firing_rates = \
    fr_funcs.get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event=avg_firing_rates[trial_type],
                                                           time_around_pattern=time_before_and_after_event,
                                                           pattern_regions_to_compare=[0, 0.8, 1.0, 1.2],
                                                           comparison_factor=4, comparison_direction='decrease',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(decreasing_firing_rates)

# Show where the neurons are in the brain
template_info_decreasing_fr_neurons = template_info.iloc[decreasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const_comm.PROBE_DIMENSIONS,
                                     const_comm.POSITION_MULT, template_info=template_info_decreasing_fr_neurons)

pd.to_pickle(template_info_decreasing_fr_neurons, join(poke_folder, 'ti_decreasing_neurons_on_trial_pokes.df'))

# Have a look at the neuron with the largest decrease
time_before_and_after_event = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
frames_around_beam_break = 120 * time_before_and_after_event
args = [decreasing_firing_rates_neuron_index, avg_firing_rates[trial_type], template_info, spike_info,
        trials[trial_type], frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(decreasing_firing_rates_neuron_index) - 1])

# </editor-fold>

'''
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
'''

# -------------------------------------------------
# <editor-fold desc="Generate raster plots for all neurons around poke event">

trial_type = 's'

all_pic_folder = join(results_folder, 'Images', 'Modulating_Neurons_On_Events', 'All_neurons')
fig = plt.figure(0, figsize=(8, 6), dpi=150)
time_before_and_after_event = 8
frames_around_beam_break = 120 * time_before_and_after_event

dec_and_inc_neurons_indices = np.union1d(decreasing_firing_rates_neuron_index, increasing_firing_rates_neuron_index)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)
for index in position_sorted_indices:
    fig = fr_funcs.show_rasters(index, template_info, spike_info,
                                trials[trial_type], frames_around_beam_break, fig)
    template_number = template_info.iloc[index]['template number']
    firing_rate = "%3.3f" % template_info.iloc[index]['firing rate']
    y_position = int(template_info.iloc[index]['position Y'] * const_comm.POSITION_MULT)
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

trial_type = 'ns'
time_before_and_after_event = 0.1
timepoints_around_beam_break = int(const_comm.SAMPLING_FREQUENCY * time_before_and_after_event)

voltage_around_trials = np.empty((len(trials[trial_type]), const_comm.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                                  2 * timepoints_around_beam_break))
for st in np.arange(len(trials[trial_type])):
    voltage_around_trials[st, :, :] = raw_data[:, trials[trial_type][st] - timepoints_around_beam_break:
                                                  trials[trial_type][st] + timepoints_around_beam_break]

avg_voltage_around_trials = np.mean(voltage_around_trials, axis=0)

_ = plt.imshow(np.flipud(avg_voltage_around_trials), aspect='auto')
# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="LOOK AT ALL THE NEURONS AROUND THE POKE EVENT">

trial_type = 'r'

smooth_time = 0.5
smooth_frames = smooth_time * 120

data = avg_firing_rates[trial_type]
events = {'s':'Successful', 'ns': 'Not successful', 'tb': 'Ball Touch', 'r': 'Random'}
frs = binning.rolling_window_with_step(data, np.mean, smooth_frames, int(smooth_frames / 3))

frs_norm = np.asarray(frs)
for i in np.arange(len(frs)):
    frs_norm[i, :] = binning.scale(frs[i], 0, 1)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)

regions_pos = list(const.BRAIN_REGIONS.values())
region_lines = []
for rp in regions_pos:
    region_lines.append(sync_funcs.find_nearest(y_positions[position_sorted_indices] * const_comm.POSITION_MULT, rp)[0])
region_lines = np.array(region_lines)

frs_norm_sorted = frs_norm[position_sorted_indices]

'''
plt.imshow(np.flipud(frs_norm_sorted), aspect='auto')
plt.hlines(y=len(frs) - region_lines, xmin=0, xmax=len(frs_norm_sorted[0])-1, linewidth=3, color='w')
plt.vlines(x=int(len(frs_norm_sorted[0]) / 2), ymin=0, ymax=len(frs_norm_sorted) - 1)
'''

plt.imshow(np.flipud(frs_norm_sorted), aspect='auto', extent=[-8, 8, len(frs_norm_sorted), 0])
plt.hlines(y=len(frs_norm_sorted) - region_lines, xmin=-8, xmax=8, linewidth=2, color='w')
plt.vlines(x=0, ymin=0, ymax=len(frs_norm_sorted) - 1)
plt.title('Rat = {}, Day from Imp. = {}, Event = {}, Trials = {}'\
          .format(const.rat_folder[3:], str(date_folder), events[trial_type],
                  str(len(trials[trial_type]))))
plt.tight_layout()

i = 0
sv.graph_pane(globals(), 'i', 'tn')


time_before_and_after_event = 8
index = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)
output = None
all_indices = np.arange(len(avg_firing_rates[trial_type]))
frames_around_beam_break = 120 * time_before_and_after_event
args = [all_indices, avg_firing_rates[trial_type], template_info, spike_info,
        trials[trial_type], frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters_for_live_update

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_decrease', 'args',
                    slider_limits=[0, len(avg_firing_rates[trial_type]) - 1])
# </editor-fold>
