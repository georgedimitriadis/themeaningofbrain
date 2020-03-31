

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)

camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="EVENT TIMES OF REWARD SOUNDS">

sounds_dur = sounds[:, 1] - sounds[:, 0]
reward_sounds = sounds[sounds_dur < 4000]

reward_sounds = reward_sounds[:, 0]

# np.save(join(events_definitions_folder, 'events_reward_sounds.npy'), reward_sounds)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="EVENT TIMES OF POKES THAT WERE THE END OF A SUC. TRIAL (FIRST POKE <40SEC AFTER A TOUCH BALL)">

all_pokes = beam_breaks[:, 0]
successful_trial_pokes = []
for trial in reward_sounds:
    index = np.argmin(np.abs(all_pokes - trial))
    successful_trial_pokes.append(all_pokes[index])
successful_trial_pokes = np.array(successful_trial_pokes)

# np.save(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy'), successful_trial_pokes)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="SANITY CHECK THAT THE POKE EVENTS MAKE SENSE">

sync_window = 0.5
sync_window_timepoints = int(sync_window * const.SAMPLING_FREQUENCY)

sync_around_trial_pokes = np.empty((len(successful_trial_pokes), 2 * sync_window_timepoints))
for st in np.arange(len(successful_trial_pokes)):
    sync_around_trial_pokes[st, :] = sync[successful_trial_pokes[st] - sync_window_timepoints:
                                    successful_trial_pokes[st] + sync_window_timepoints]

f = plt.figure(0)
_ = plt.plot(np.mean(sync_around_trial_pokes, axis=0))
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="EVENT TIMES OF POKES THAT WERE NOT PART OF A TRIAL (FIRST POKES AFTER A PERIOD OF NOT POKING)">

minimum_delay = 5
time_to_previous_poke = np.diff(beam_breaks[:, 0])/const.SAMPLING_FREQUENCY
pokes_after_delay = np.squeeze(beam_breaks[np.argwhere(time_to_previous_poke > minimum_delay) + 1])
first_pokes_after_delay = pokes_after_delay[:, 0]

# Check if any of the pokes after delay are also in the spaces between touching the ball and the reward sound
# playing and remove those

overlaps = []
for r_sound in successful_trial_pokes:
    t = np.logical_and(first_pokes_after_delay > r_sound - 2 * const.SAMPLING_FREQUENCY,
                       first_pokes_after_delay < r_sound + 2 * const.SAMPLING_FREQUENCY)
    if np.any(t):
        overlaps.append(np.squeeze(np.argwhere(t)))

overlaps = np.squeeze(np.array(overlaps))
print(overlaps)
print(len(overlaps))

first_pokes_after_delay = np.delete(first_pokes_after_delay, overlaps)
if date_folder == 8:
    first_pokes_after_delay = first_pokes_after_delay[1:]

# np.save(join(events_definitions_folder, 'events_first_pokes_after_{}_delay_non_reward.npy'.format(str(minimum_delay)))
#         , first_pokes_after_delay)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SANITY CHECK THAT THE POKE EVENTS MAKE SENSE">

sync_window = 0.5
sync_window_timepoints = int(sync_window * const.SAMPLING_FREQUENCY)

sync_around_non_trial_pokes = np.empty((len(first_pokes_after_delay), 2 * sync_window_timepoints))
for st in np.arange(len(first_pokes_after_delay)):
    sync_around_non_trial_pokes[st, :] = sync[first_pokes_after_delay[st] - sync_window_timepoints:
                                    first_pokes_after_delay[st] + sync_window_timepoints]

_ = plt.plot(np.mean(sync_around_non_trial_pokes, axis=0))
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="EVENT TIMES OF TOUCH BALLS">
events_touch_ball = event_dataframes['ev_rat_touch_ball']['AmpTimePoints'].values

events_touch_ball_successful_trial = []
for tb_event in events_touch_ball:
    suc_trial_poke_after_tb_event = successful_trial_pokes[np.where(successful_trial_pokes > tb_event)[0][0]]
    if suc_trial_poke_after_tb_event - tb_event < 40 * const.SAMPLING_FREQUENCY:
        events_touch_ball_successful_trial.append(tb_event)
events_touch_ball_successful_trial = np.array(events_touch_ball_successful_trial)

# np.save(join(events_definitions_folder, 'events_touch_ball.npy'), events_touch_ball)
# np.save(join(events_definitions_folder, 'events_touch_ball_successful_trial.npy'), events_touch_ball_successful_trial)

# </editor-fold>
# -------------------------------------------------


