


from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded import \
    constants_common as const_comm
from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._47p2 \
    import constants_47p2 as const_rat
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis.LFP import emd
from mne.time_frequency import multitaper as mt

import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import hilbert

import pickle


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date = 8

data_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                            'Data')

analysis_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                       'Analysis', 'NeuropixelSimulations', 'Sparce')
kilosort_folder = join(analysis_folder, 'Kilosort')

results_folder = join(analysis_folder, 'Results')
spike_lfp_folder = join(results_folder, 'SpikeLfpCorrelations', 'SpikesAwayFromSuccTrials')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))


imfs_file = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                       'Analysis', 'Lfp', 'EMD', 'imfs.bin')
imfs = emd.load_memmaped_imfs(imfs_file, const_comm.NUMBER_OF_IMFS, const_comm.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="GET TIMES OF SUCCESSFUL TRIALS">

camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const_comm.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
sounds_dur = sounds[:, 1] - sounds[:, 0]
reward_sounds = sounds[sounds_dur < 4000]

# Using the start of the reward tone to generate events
# There is a difference of 78.6 frames (+-2) between the reward tone and the csv file event (about 700ms)
succesful_trials = reward_sounds[:, 0]
# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="DEFINE THE TRIAL WINDOW">

start_offset = 0 * const_comm.SAMPLING_FREQUENCY
trials_start = succesful_trials + start_offset
trials_window_size = 1.0 * const_comm.SAMPLING_FREQUENCY

imf_subsampling_factor = const_comm.LFP_DOWNSAMPLE_FACTOR

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="CREATE SPIKE TIMES OF ALL NEURONS FOR TRIAL WINDOWS">

spike_times_in_trials_windows = {}
for neuron in np.arange(len(template_info)):
    temp = np.empty(0)
    spike_times = spike_info[np.isin(spike_info['original_index'],
                                     template_info.iloc[neuron]['spikes in template'])]['times']
    for start in trials_start:
        spike_times_in_window = np.array(spike_times)[np.logical_and(spike_times > start,
                                                                     spike_times < start + trials_window_size)]
        if len(spike_times_in_window) > 0:
            temp = np.concatenate((temp, spike_times_in_window))
    if len(temp) > 0:
        spike_times_in_trials_windows[neuron] = temp
    print('Done neuron {} with {} spikes'.format(neuron, len(temp)))

with open(join(results_folder, 'EventsCorrelations', 'Poke', 'spike_times_between_{}_and_{}_succ_trials.pcl'.format(start_offset, trials_window_size)),
          'wb') as f:
    pickle.dump(spike_times_in_trials_windows, f)

# </editor-fold>


with open(join(results_folder, 'EventsCorrelations', 'Poke', 'spike_times_between_{}_and_{}_succ_trials.pcl'.format(start_offset, trials_window_size)),
          'rb') as f:
    spike_times_in_trials_windows = pickle.load(f)


# -------------------------------------------------
# <editor-fold desc="CREATE SPIKES BINARY MATRIX FOR ALL NEURONS AT THE SUBSAMPLED RATE">

def spikes_matrix(_start_offset, _trials_window_size):
    _start_offset = _start_offset * const_comm.SAMPLING_FREQUENCY
    _trials_window_size = _trials_window_size * const_comm.SAMPLING_FREQUENCY

    trials_start = succesful_trials + _start_offset
    step = int(_trials_window_size / imf_subsampling_factor)
    spikes_in_trials_windows = np.empty((len(template_info), int(len(trials_start) * step)))
    for neuron in np.arange(len(template_info)):
        spike_times = spike_info[np.isin(spike_info['original_index'],
                                         template_info.iloc[neuron]['spikes in template'])]['times']

        for s in np.arange(len(trials_start)):
            start = trials_start[s]
            spike_times_in_window = np.array(spike_times)[np.logical_and(spike_times > start,
                                                                         spike_times < start + _trials_window_size)]
            temp = np.zeros(step)
            indices = ((spike_times_in_window - start) / imf_subsampling_factor).astype(np.int)
            temp[indices] = 1
            spikes_in_trials_windows[neuron, s * step : (s + 1) * step] = temp

        print('Done neuron {}'.format(neuron))

    with open(join(results_folder, 'EventsCorrelations', 'Poke',
                   'spike_binary_matrix_subsampled_between_{}_and_{}_succ_trials.pcl'.
                           format(_start_offset / const_comm.SAMPLING_FREQUENCY,
                                  (_start_offset + _trials_window_size) / const_comm.SAMPLING_FREQUENCY)),
              'wb') as f:
        pickle.dump(spikes_in_trials_windows, f)

    return spikes_in_trials_windows


# Run for a specific window
spikes_in_trials_windows = spikes_matrix(_start_offset=start_offset, _trials_window_size=1)

# OR Create spike matrices for many times around the event
for s in np.arange(-40, 40, 2):
    _ = spikes_matrix(_start_offset=s, _trials_window_size=1)
    print('FINISHED OFFSET {}'.format(s))
    print('--------------------')

# </editor-fold>


with open(join(results_folder, 'EventsCorrelations', 'Poke',
               'spike_binary_matrix_subsampled_between_{}_and_{}_succ_trials.pcl'.
                       format(start_offset / const_comm.SAMPLING_FREQUENCY,
                              (start_offset + trials_window_size) / const_comm.SAMPLING_FREQUENCY)),
          'rb') as f:
    spikes_in_trials_windows = pickle.load(f)


# -------------------------------------------------
# <editor-fold desc="CREATE RANDOM SPIKING DATASET">

random_spikes_in_trials_windows = np.copy(spikes_in_trials_windows)
random_spikes_in_trials_windows = random_spikes_in_trials_windows.T
np.random.shuffle(random_spikes_in_trials_windows)
random_spikes_in_trials_windows = random_spikes_in_trials_windows.T

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="GET INSTANTANEOUS PHASES FOR TRIAL WINDOWS">

imf_trial_starts = (trials_start / imf_subsampling_factor).astype(np.int)
imf_trials_window_size = int(trials_window_size / imf_subsampling_factor)

imfs_in_trial_windows = imfs[:, :, imf_trial_starts[0]:imf_trial_starts[0] + imf_trials_window_size]
for start in imf_trial_starts[1:]:
    imfs_in_window = imfs[:, :, start:start + imf_trials_window_size]
    imfs_in_trial_windows = np.concatenate((imfs_in_trial_windows, imfs_in_window), axis=-1)

analytic_signal = hilbert(imfs_in_trial_windows)
instant_phi = np.angle(analytic_signal)

modular_phi = np.copy(instant_phi)
modular_phi[np.logical_or(instant_phi >= 2, instant_phi < -2)] = 4
modular_phi[np.logical_and(instant_phi < 2, instant_phi >= 0.75)] = 3
modular_phi[np.logical_and(instant_phi < 0.75, instant_phi >= -0.75)] = 2
modular_phi[np.logical_and(instant_phi < -0.75, instant_phi >= -2)] = 1

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="FIND AVERAGE PHASE LOCKING VALUES">

average_number_of_spikes_per_trial_threshold = 2

good_lfp_channels = np.arange(imfs.shape[0])
good_lfp_channels = np.delete(good_lfp_channels, np.arange(55, 60))

average_phase = []
std_phase = []

average_random_phase = []
std_random_phase = []

average_modular = []
std_modular = []

average_amplitude = []
std_amplitude = []

neurons_used = []

for neuron in np.arange(spikes_in_trials_windows.shape[0]):
    spike_indices = np.squeeze(np.argwhere(spikes_in_trials_windows[neuron] == 1))
    random_spike_indices = np.squeeze(np.argwhere(random_spikes_in_trials_windows[neuron] == 1))
    try:
        spike_index_length = len(spike_indices)
    except:
        spike_index_length = 0
    if spike_index_length >= len(trials_start) * average_number_of_spikes_per_trial_threshold:
        phases = instant_phi[:, :, spike_indices]
        random_phases = instant_phi[:, :, random_spike_indices]
        modular = modular_phi[:, :, spike_indices]
        amplitude = imfs_in_trial_windows[:, :, spike_indices]
        if len(phases.shape) == 2:
            phases = np.expand_dims(phases, axis=-1)

        average_phase.append(np.mean(phases, axis=-1))
        std_phase.append(np.std(phases, axis=-1))

        average_random_phase.append(np.mean(random_phases, axis=-1))
        std_random_phase.append(np.std(random_phases, axis=-1))

        average_modular.append(np.mean(modular, axis=-1))
        std_modular.append(np.std(modular, axis=-1))

        average_amplitude.append(np.mean(amplitude, axis=-1))
        std_amplitude.append(np.std(amplitude, axis=-1))

        neurons_used.append(neuron)

average_phase = np.array(average_phase)[:, good_lfp_channels, :]
std_phase = np.array(std_phase)[:, good_lfp_channels, :]

average_random_phase = np.array(average_random_phase)[:, good_lfp_channels, :]
std_random_phase = np.array(std_random_phase)[:, good_lfp_channels, :]

average_modular = np.array(average_modular)[:, good_lfp_channels, :]
std_modular = np.array(std_modular)[:, good_lfp_channels, :]

average_amplitude = np.array(average_amplitude)[:, good_lfp_channels, :]
std_amplitude = np.array(std_amplitude)[:, good_lfp_channels, :]

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="ARRANGE NEURONS BY DEPTH">

height_of_used_neurons = template_info.iloc[neurons_used]['position Y'].values
position_sorted_indices = np.argsort(height_of_used_neurons)

average_phase_height_sorted = average_phase[position_sorted_indices, :, :]
std_phase_height_sorted = std_phase[position_sorted_indices, :, :]

average_random_phase_height_sorted = average_random_phase[position_sorted_indices, :, :]
std_random_phase_height_sorted = std_random_phase[position_sorted_indices, :, :]

average_modular_height_sorted = average_modular[position_sorted_indices, :, :]
std_modular_height_sorted = std_modular[position_sorted_indices, :, :]

average_amplitude_height_sorted = average_amplitude[position_sorted_indices, :, :]
std_amplitude_height_sorted = std_amplitude[position_sorted_indices, :, :]

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="VISUALISE">
psd_imf, fs = mt.psd_array_multitaper(imfs[:, :, :20000],
                                      sfreq=const_comm.SAMPLING_FREQUENCY/imf_subsampling_factor,
                                      fmin=1, fmax=3000, bandwidth=6, verbose=0)

peak_freqs = fs[np.argmax(psd_imf[:, :, :], axis=2)]
peak_freqs = np.mean(peak_freqs[:2, :], axis=0)


plt.figure(40)
plot_indices = np.arange(0, 12)
for i in plot_indices:
    plt.subplot(len(plot_indices), 1, i+1)
    plt.plot(fs, psd_imf[40, i, :])


show = [average_phase_height_sorted, std_phase_height_sorted]
min_imf_to_show = 13
plt.figure(1)
plot_indices = np.arange(0, min_imf_to_show)
for i in plot_indices:
    plt.subplot(1, len(plot_indices), i+1)
    plt.imshow(show[0][:, :, i], vmin=1.1*show[0].min(), vmax=1.1*show[0].max()) # modular = 0.2 / 1.5
    plt.title('{} Hz'.format(peak_freqs[i]))

plt.figure(2)
plot_indices = np.arange(0, min_imf_to_show)
for i in plot_indices:
    plt.subplot(1, len(plot_indices), i+1)
    plt.imshow(show[1][:, :, i], vmin=-1.1*show[1].min(), vmax=1.1*show[1].max()) # modular = 1.3 / 4
    plt.title('{} Hz'.format(peak_freqs[i]))

plt.figure()
plt.imshow(show[1][:, :, 2], vmin=-1.1*show[1].min(), vmax=1.1*show[1].max()) # modular = 1.3 / 4
plt.title('{} Hz'.format(peak_freqs[2]))

f, (a1, a2) = plt.subplots(2, 1, sharex=True, sharey=False)
a1.plot(imfs_in_trial_windows[20, 5, :1000])
a2.plot(instant_phi[20, 5, :1000])
# </editor-fold>


