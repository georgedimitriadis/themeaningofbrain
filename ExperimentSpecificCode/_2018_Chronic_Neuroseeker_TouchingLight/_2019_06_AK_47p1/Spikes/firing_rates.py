
from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions import events_sync_funcs as \
    sync_funcs
from BrainDataAnalysis import binning

import pandas as pd


# -------------------------------------------------
# LOAD FOLDERS
# -------------------------------------------------
date_folder = 6

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

events_folder = join(data_folder, "events")


# CALCULATE SPIKING RATES
# Make the spike rates using each frame as a binning window

#  Load the pre generated DataFrames for the event CSVs
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))
sampling_frequency = const.SAMPLING_FREQUENCY
spike_rates = binning.spike_count_per_frame(template_info, spike_info, event_dataframes['ev_video'],
                                            sampling_frequency, file_to_save_to=file_to_save_to)

# Using the frame based spikes rates do a rolling window to average a bit more
num_of_frames_to_average = 0.25/(1/120)

spike_rates_0p25 = []
for n in np.arange(spike_rates.shape[0]):
    spike_rates_0p25.append(binning.rolling_window_with_step(spike_rates[n, :], np.mean,
                                                             num_of_frames_to_average, num_of_frames_to_average))
spike_rates_0p25 = np.array(spike_rates_0p25)
np.save(join(kilosort_folder, 'firing_rate_with_0p25s_window.npy'), spike_rates_0p25)


