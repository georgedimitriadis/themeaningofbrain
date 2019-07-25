


from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp

import pandas as pd
import matplotlib.pyplot as plt

import common_data_transforms as cdt
import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Denoised', 'Kilosort')

events_folder = join(data_folder, "events")

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

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
minimum_delay = 5
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

time_around_beam_break = 6
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