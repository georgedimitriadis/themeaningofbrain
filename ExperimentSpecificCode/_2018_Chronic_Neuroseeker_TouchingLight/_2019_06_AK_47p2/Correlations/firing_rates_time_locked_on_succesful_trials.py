

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp


import pandas as pd
import matplotlib.pyplot as plt

import common_data_transforms as cdt
import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

events_folder = join(data_folder, "events")

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
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
time_around_beam_break = 6
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
                                                           pattern_regions_to_compare=[0, 0.6, 0.8, 1.2],
                                                           comparison_factor=3, comparison_direction='increase',
                                                           baserate=0.1)

fr_funcs.show_firing_rates_around_event(increasing_firing_rates)

# Show where the neurons are in the brain
template_info_increasing_fr_neurons = template_info.iloc[increasing_firing_rates_neuron_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info_increasing_fr_neurons)

# Have a detailed look at the neuron with the largest increase
# largest_increase_neuron_index = increasing_firing_rates_neuron_index[np.argmax(increasing_firing_rates_ratio)]
index = 0
fig1 = plt.figure(0)
fig2 = plt.figure(1)
output = None
frames_around_beam_break = 120 * time_around_beam_break
time_points_around_beam_break = const.SAMPLING_FREQUENCY * time_around_beam_break
args = [increasing_firing_rates_neuron_index, avg_firing_rate_around_suc_trials, template_info, spike_info,
        succesful_trials, frames_around_beam_break, fig1, fig2]

show_rasters_increase = fr_funcs.show_rasters

sl.connect_repl_var(globals(), 'index', 'output', 'show_rasters_increase', 'args',
                    slider_limits=[0, len(increasing_firing_rates_neuron_index) - 1])

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="NEURONS THAT DECREASE THEIR FIRING RATES AROUND THE POKE EVENT">
# Find which neurons decrease their firing rate on average around a successful poke
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

# Have a look at the neuron with the largest decrease
index = 0
fig1 = plt.figure(0)
fig2 = plt.figure(1)
output = None
frames_around_beam_break = 120 * time_around_beam_break
args = [decreasing_firing_rates_neuron_index, avg_firing_rate_around_suc_trials, template_info, spike_info,
        succesful_trials, frames_around_beam_break, fig1, fig2]

show_rasters_decrease = fr_funcs.show_rasters

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



