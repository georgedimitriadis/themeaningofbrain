

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import pandas as pd
from scipy import io


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')

events_folder = join(data_folder, "events")

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')
event_definition_folder = join(results_folder, 'EventsDefinitions')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="CREATE THE SPIKE TIME MATRIX BUT ONLY FOR A FEW SECONDS AROUND THE BEAM BREAK EVENTS">

start_pokes_after_delay = np.load(join(event_definition_folder, 'events_first_pokes_after_5_delay_non_reward.npy'))
start_pokes_after_delay = start_pokes_after_delay[1:]

time_around_beam_break = 3
min_max_timepoints = np.array([start_pokes_after_delay - time_around_beam_break * const.SAMPLING_FREQUENCY,
                              start_pokes_after_delay + time_around_beam_break * const.SAMPLING_FREQUENCY]).T

print(np.min(min_max_timepoints[1:, 0] - min_max_timepoints[:-1, 1]))  # Check that the regions do not overlap

all_cell_spikes_around_beam_break = []
for c in np.arange(len(template_info)):
    cell_spikes = np.empty(1)
    for tp in min_max_timepoints:
        t = template_info.iloc[c]['spikes in template']
        t1 = t[tp[0] < t]
        t2 = np.array(t1[tp[1] > t1])
        cell_spikes = np.concatenate((cell_spikes, t2))
    cell_spikes = cell_spikes[1:] / const.SAMPLING_FREQUENCY
    all_cell_spikes_around_beam_break.append(cell_spikes)

max_time_series_length = np.max([len(c) for c in all_cell_spikes_around_beam_break])
# OR
max_time_series_length = 7000
all_cell_spikes_around_beam_break_array = np.empty((len(all_cell_spikes_around_beam_break), max_time_series_length))
last_spikes = []
for i, c in enumerate(all_cell_spikes_around_beam_break):
    temp_len = len(c)
    all_cell_spikes_around_beam_break_array[i, :temp_len] = c
    all_cell_spikes_around_beam_break_array[i, temp_len:] = [np.nan] * (max_time_series_length - temp_len)
    if len(c) > 0:
        last_spikes.append(c[-1])
    else:
        last_spikes.append(0)

sub_array = all_cell_spikes_around_beam_break_array[np.random.choice(np.arange(all_cell_spikes_around_beam_break_array.shape[0]), 50, replace=False),
                                                    :7000]
io.savemat(r'E:\Code\Others\Cell-Assembly-Detection\Programs_and_data\non_rewarded_beam_breaks.mat',
           {'spM': sub_array})
# </editor-fold>
