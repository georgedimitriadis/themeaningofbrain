
"""
The pipeline for spikesorting this dataset
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing as preproc

from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from spikesorting_tsne_guis import clean_kilosort_templates as clean
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._47p1 \
    import constants_47p1 as const_rat
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded \
    import constants_common as const_com

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions import events_sync_funcs as \
    sync_funcs, firing_rates_sync_around_events_funcs as fr_funcs
from BrainDataAnalysis.Statistics import binning

import common_data_transforms as cdt
import sequence_viewer as sv
import slider as sl

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "FOLDERS NAMES"
date = 6
binary_data_filename = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
analysis_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                       'Analysis')
kilosort_folder = join(analysis_folder, 'NeuropixelSimulations', 'Sparce', 'Kilosort')

tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

data_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date], 'Data')
events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)

results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
events_touch_ball = np.load(join(events_definitions_folder, 'events_touch_ball.npy'))

sampling_freq = const_com.SAMPLING_FREQUENCY

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#  STEP 1: RUN KILOSORT ON THE DATA
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "STEP 2: CLEAN SPIKESORT (RIGHT AFTER KILOSORT)"

# a) Create average of templates:
# To create averages of templates use cmd (because the create_data_cubes doesn't work when called from a REPL):
# Go to where the create_data_cubes.py is (in spikesort_tsne_guis/spikesort_tsen_guis) and run the following python command
# (you can use either the raw or the denoised data to create the average)
# python E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis\create_data_cubes.py
#                                                                                original
#                                                                                "F:\Neuroseeker chronic\AK_47.2\2019_06_25-12_50\Analysis\NeuropixelSimulations\Long\Kilosort"
#                                                                                "F:\Neuroseeker chronic\AK_47.2\2019_06_25-12_50\Data\Amplifier_APs.bin"
#                                                                                1368
#                                                                                50
# (Use single space between parameters, not Enter like here)
# (Change the folders as appropriate for where the data is)

# b) Clean:
clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=const_com.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const_com.prb_file,
                              type_of_binary=const_com.BINARY_FILE_ENCODING,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)

# c) Remove some types
template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))
print(len(np.argwhere(template_marking == 0)))
print(len(np.argwhere(template_marking == 1)))
print(len(np.argwhere(template_marking == 2)))
print(len(np.argwhere(template_marking == 3)))
print(len(np.argwhere(template_marking == 4)))
print(len(np.argwhere(template_marking == 5)))
print(len(np.argwhere(template_marking == 6)))
print(len(np.argwhere(template_marking == 7)))
template_marking[np.argwhere(template_marking == 5)] = 0
template_marking[np.argwhere(template_marking == 6)] = 0
np.save(join(kilosort_folder, 'template_marking.npy'), template_marking)
# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "STEP 3: CREATE TEMPLATE INFO OF ALL THE CLEANED TEMPLATES"

# a) Create the positions of the templates on the probe (and have a look)
_ = spp.generate_probe_positions_of_templates(kilosort_folder)
bad_channel_positions = spp.get_y_spread_regions_of_bad_channel_groups(kilosort_folder, const_rat.bad_channels)
spp.view_grouped_templates_positions(kilosort_folder, const_rat.BRAIN_REGIONS, const_com.PROBE_DIMENSIONS,
                                     const_com.POSITION_MULT)

# b) Create the template_info.df dataframe (or load it if you already have it)
# template_info = preproc_kilo.generate_template_info_after_cleaning(kilosort_folder, sampling_freq)
template_info = np.load(join(kilosort_folder, 'template_info.df'), allow_pickle=True)

# c) Make the spike info from the initial, cleaned, kilosort results
# spike_info = preproc_kilo.generate_spike_info_after_cleaning(kilosort_folder)
spike_info = np.load(join(kilosort_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
spp.view_grouped_templates_positions(kilosort_folder, const_rat.BRAIN_REGIONS, const_com.PROBE_DIMENSIONS,
                                     const_com.POSITION_MULT, template_info=template_info)
# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "CALCULATE SPIKING RATES">
# Make the spike rates using each frame as a binning window

#  Load the pre generated DataFrames for the event CSVs
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))
spike_rates = binning.spike_count_per_frame(template_info, spike_info, event_dataframes['ev_video'],
                                            sampling_freq, file_to_save_to=file_to_save_to)

# Using the frame based spikes rates do a rolling window to average a bit more
num_of_frames_to_average = 0.25/(1/120)

spike_rates_0p25 = []
for n in np.arange(spike_rates.shape[0]):
    spike_rates_0p25.append(binning.rolling_window_with_step(spike_rates[n, :], np.mean,
                                                             num_of_frames_to_average, num_of_frames_to_average))
spike_rates_0p25 = np.array(spike_rates_0p25)
np.save(join(kilosort_folder, 'firing_rate_with_0p25s_window.npy'), spike_rates_0p25)
# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="GET TIMES AND FRAMES OF SUCCESSFUL TRIALS">

video_frame_spike_rates_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const_com.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
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

# Get the average firing rates of all neurons a few seconds around the touch ball
time_around_beam_break = 8
avg_firing_rate_around_touch_ball = fr_funcs.get_avg_firing_rates_around_events(spike_rates=spike_rates,
                                                                                event_time_points=events_touch_ball,
                                                                                ev_video_df=event_dataframes['ev_video'],
                                                                                time_around_event=time_around_beam_break)
# </editor-fold>

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc="LOOK AT ALL THE NEURONS AROUND THE POKE EVENT">

smooth_time = 0.5
smooth_frames = smooth_time * 120

t = binning.rolling_window_with_step(avg_firing_rate_around_touch_ball, np.mean, smooth_frames, int(smooth_frames / 3))
tn = preproc.normalize(t, norm='l1', axis=0)

tn = np.asarray(t)
for i in np.arange(len(t)):
    tn[i, :] = binning.scale(t[i], 0, 1)

y_positions = template_info['position Y'].values
position_sorted_indices = np.argsort(y_positions)

regions_pos = list(const_rat.BRAIN_REGIONS.values())
region_lines = []
for rp in regions_pos:
    region_lines.append(sync_funcs.find_nearest(y_positions[position_sorted_indices] * const_com.POSITION_MULT, rp)[0])
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
# ----------------------------------------------------------------------------------------------------------------------


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
             'mutual_infos_spike_rates_vs_spike_rates_all_neurons_around_succ_trials.npy'), mutual_infos)
