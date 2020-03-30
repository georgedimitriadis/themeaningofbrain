
from os.path import join
import numpy as np

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._47p2 \
    import constants_47p2 as const
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded \
    import constants_common as const_comm
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis import timelocked_analysis_functions as tla_funcs

import matplotlib.pyplot as plt


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
# Folder definitions
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')
lfp_average_data_folder = join(results_folder, 'Lfp', 'Averages')

ap_data_filename = join(data_folder, 'Amplifier_APs.bin')

# Load data

ap_data = ns_funcs.load_binary_amplifier_data(ap_data_filename, const_comm.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

trials = {'s': np.load(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy')),
          'tb': np.load(join(events_definitions_folder, 'events_touch_ball.npy'))}

if date_folder != 6:
    minimum_delay = 5
    nst = np.load(join(events_definitions_folder,
                                       'events_first_pokes_after_{}_delay_non_reward.npy'.format(str(minimum_delay))))
    if date_folder == 8:
        nst = nst[1:]

    trials['ns'] = nst

trials['r'] = np.random.choice(np.arange(200000, ap_data.shape[1]-200000), len(trials['s']))


window_time = 8
window_timepoints = int(window_time * const_comm.SAMPLING_FREQUENCY)
window_downsampled = int(window_timepoints / const_comm.LFP_DOWNSAMPLE_FACTOR)

lfp_probe_positions = np.empty(const_comm.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)
lfp_probe_positions[np.arange(0, 72, 2)] = (((np.arange(9, 1440, 40) + 1) / 4).astype(np.int) + 1) * 22.5
lfp_probe_positions[np.arange(1, 72, 2)] = (((np.arange(29, 1440, 40) + 1) / 4).astype(np.int)) * 22.5
# </editor-fold>
# -------------------------------------------------

#  Make and save the MUAe
event_choises = ['s', 'tb', 'r']
if date_folder != 6:
    event_choises.append('ns')
for choise in event_choises:
    events = trials[choise]
    num_of_events = len(events)

    avg_muae_around_event, std_muae_around_event, time_axis = \
        tla_funcs.time_lock_raw_data(ap_data, events, times_to_cut=[-window_time, window_time],
                                     sampling_freq=const_comm.SAMPLING_FREQUENCY,
                                     baseline_time=[-window_time, -0.5 * window_time], sub_sample_freq=300,
                                     high_pass_cutoff=3000, rectify=True, low_pass_cutoff=400,
                                     avg_reref=True, keep_trials=False)

    np.save(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_{}.npy'.format(choise)), avg_muae_around_event)


#  Normalise
trial_type = 'r'

events = {'s':'Successful', 'ns': 'Not successful', 'tb': 'Ball Touch', 'r': 'Random'}
avg_muae_around_event = np.load(join(results_folder, 'Lfp', 'Averages', 'Muaes_around_{}.npy'.format(trial_type)))

regions_pos = np.array(list(const.BRAIN_REGIONS.values()))
pos_to_elect_factor = const_comm.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE / 8100
region_lines = binning.scale(regions_pos, np.min(regions_pos) * pos_to_elect_factor, np.max(regions_pos) * pos_to_elect_factor)

muae_smooth = binning.rolling_window_with_step(avg_muae_around_event, np.mean, 40, 40)
muae_smooth = (muae_smooth - muae_smooth.min()) / (muae_smooth.max() - muae_smooth.min())
_ = plt.figure(1)
plt.imshow(np.flipud(muae_smooth), aspect='auto', extent=[-8, 8, len(muae_smooth), 0])
plt.vlines(x=0, ymin=0, ymax=muae_smooth.shape[0] - 1)
plt.hlines(y=muae_smooth.shape[0] - region_lines, xmin=-8, xmax=8, linewidth=3, color='w')
plt.title('Rat = {}, Day from Imp. = {}, Event = {}, Trials = {}'\
          .format(const.rat_folder[3:], str(date_folder), events[trial_type],
                  str(len(trials[trial_type]))))

plt.tight_layout()


muae_norm = np.empty((avg_muae_around_event.shape))
for i in np.arange(len(avg_muae_around_event)):
    muae_norm[i, :] = binning.scale(avg_muae_around_event[i], 0, 1)

_= plt.figure(2)
plt.imshow(np.flipud(muae_norm), aspect='auto')
plt.vlines(x=muae_norm.shape[1] / 2, ymin=0, ymax=muae_norm.shape[0] - 1)
plt.hlines(y=muae_norm.shape[0] - region_lines, xmin=0, xmax=muae_norm.shape[1] - 1, linewidth=1, color='w')

muae_norm_smooth = binning.rolling_window_with_step(muae_norm, np.mean, 40, 40)

_= plt.figure(3)
plt.imshow(np.flipud(muae_norm_smooth), aspect='auto')
plt.vlines(x=muae_norm_smooth.shape[1] / 2, ymin=0, ymax=muae_norm_smooth.shape[0] - 1)
plt.hlines(y=muae_norm_smooth.shape[0] - region_lines, xmin=0, xmax=muae_norm_smooth.shape[1] - 1, linewidth=1, color='w')

_= plt.figure(3)
plt.imshow(np.flipud(muae_norm_smooth), aspect='auto', extent=[-8, 8, len(muae_norm_smooth), 0])
plt.hlines(y=muae_norm_smooth.shape[0] - region_lines, xmin=-8, xmax=8, linewidth=3, color='w')
plt.vlines(x=0, ymin=0, ymax=muae_norm_smooth.shape[0] - 1)
plt.title('Rat = {}, Day from Imp. = {}, Event = {}, Trials = {}'\
          .format(const.rat_folder[3:], str(date_folder), events[trial_type],
                  str(len(trials[trial_type]))))

plt.tight_layout()
