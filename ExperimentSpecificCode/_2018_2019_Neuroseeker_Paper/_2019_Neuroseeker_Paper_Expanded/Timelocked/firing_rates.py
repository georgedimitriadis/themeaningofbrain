

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._47p2 \
    import constants_47p2 as const
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded \
    import constants_common as const_comm
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions import events_sync_funcs as \
    sync_funcs
from BrainDataAnalysis.Statistics import binning

import pandas as pd


# -------------------------------------------------
# LOAD FOLDERS
# -------------------------------------------------
date_folder = 7

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
sampling_frequency = const_comm.SAMPLING_FREQUENCY
spike_rates = binning.spike_count_per_frame(template_info, spike_info, event_dataframes['ev_video'],
                                            sampling_frequency, file_to_save_to=file_to_save_to)
