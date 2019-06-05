
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy.signal as sig

import one_shot_viewer as osv
import sequence_viewer as sv
import drop_down as dd
import transform as tr
import slider as sl

from ExperimentSpecificCode._2019_EI_Anesthesia import constants as const

from scipy.io import wavfile


# NEEDS FIXING !
BRAIN_REGIONS = {'Out of Brain': 5700}

# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
brain_data_folder = r'F:\Neuroseeker_EI\2019_05_06\Data\NeuroSeeker'
events_folder = r'F:\Neuroseeker_EI\2019_05_06\Analysis\Events'

binary_data_filename = join(brain_data_folder, r'concatenated_data_before_and_after_muscimol_APs.bin')

rec1_sync_filename = join(brain_data_folder, r'2019-05-06T16_46_10_Sync.bin')
rec2_sync_filename = join(brain_data_folder, r'2019-05-06T17_41_14_Sync.bin')
full_sync_filename = join(brain_data_folder, r'concatenated_data_before_and_after_muscimol_Sync.bin')


sounds_basic_folder = r'F:\Neuroseeker_EI\2019_05_06\Data\Sounds'
rec1_part1_wav_filename = join(sounds_basic_folder, r'2019_05_06 Recording 1', r'T2019-05-06_16-47-55_001.wav')
rec1_part2_wav_filename = join(sounds_basic_folder, r'2019_05_06 Recording 1', r'T2019-05-06_17-18-20_002.wav')
rec1_both_sounds_wav_filename = join(sounds_basic_folder, r'2019_05_06 Recording 1', r'T2019-05-06_16-47-55_full.wav')
rec2_wav_filename = join(sounds_basic_folder, r'2019_05_06 Recording 2', r'T2019-05-06_17-41-22_008.wav')
# ----------------------------------------------------------------------------------------------------------------------

# SYNC STUFF
sync_sampling_freq = const.SAMPLING_FREQUENCY

sync1 = np.fromfile(rec1_sync_filename, dtype=np.uint16).astype(np.int32)
sync = np.fromfile(full_sync_filename, dtype=np.uint16).astype(np.int32)
sync -= sync.min()

sync_diff = np.diff(sync)

'''
# Have a quick look
sync_start = 2160000
sync_step = 60000
sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync')
sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync_diff')
'''

# ----------------------------------------------------------------------------------------------------------------------
# SOUND STUFF
rec1_part1 = wavfile.read(rec1_part1_wav_filename)
rec1_part2 = wavfile.read(rec1_part2_wav_filename)
rec2 = wavfile.read(rec2_wav_filename)

sound_sampling_freq = rec1_part1[0]

# wavfile.write(rec1_both_sounds_wav_filename, sound_sampling_freq, np.concatenate((rec1_part1[1], rec1_part2[1])))

rec1_sound = wavfile.read(rec1_both_sounds_wav_filename)[1]
rec2_sound = rec2[1]

'''
sound_start = 1300000
sound_step = 274000
sv.graph_range(globals(), 'sound_start', 'sound_step', 'rec1_sound')
'''

# ----------------------------------------------------------------------------------------------------------------------
# FINDING THE TIME POINTS OF SOUNDS

# Find the starting point of each sound (6 pips)
'''
sync_signal_start = 2180000
baseline = [-3, 3]
smallest_sound = 6
baseline_time = [800, 30]
times_between_pips = 2380
length_of_six_pip_sound = 12000

sound_starting_times = [0]
prev_i = 0
for i in np.arange(sync_signal_start, len(sync_diff), 1):
    if sync_diff[i] > smallest_sound and \
            (np.all(sync_diff[i - baseline_time[0]:i-baseline_time[1]] > baseline[0])
             and np.all(sync_diff[i - baseline_time[0]:i-baseline_time[1]] < baseline[1])):
        if i > prev_i + baseline_time[1]:
            if i > sound_starting_times[-1] + length_of_six_pip_sound:
                sound_starting_times.append(i)
        prev_i = i
sound_starting_times.pop(0)
sound_starting_times = np.array(sound_starting_times)

np.save(join(events_folder, 'sound_starting_times.npy'), sound_starting_times)

sound_starting_times = np.load(join(events_folder, 'sound_starting_times.npy'))
'''

# Use this to clean any mistakenly found starting points
# -------------------------


def step_back(i, step):
    return i-step


s = 1000
args = [s]
sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync_diff')
dd.connect_repl_var(globals(), 'sound_starting_times', 'step_back', 'sync_start', 'args')
# -------------------------

# Find the end points of the 6 pip sounds
# -------------------------
pip_length = 500
inter_pip_distance = 1880
times_between_pips = pip_length + inter_pip_distance
sound_starting_times = np.load(join(events_folder, 'sound_starting_times.npy'))

six_pip_sound_start_times = []
for start in sound_starting_times:
    pip = sync_diff[start:start+pip_length]
    m = 0.5 * np.mean(pip[sig.find_peaks(pip)[0]])
    check = True
    for i in range(6):
        pip = sync_diff[start + i*times_between_pips:start + i*times_between_pips + pip_length]
        if np.mean(pip[sig.find_peaks(pip)[0]]) < m:
            check = False
    if check:
        six_pip_sound_start_times.append(start)
six_pip_sound_start_times = np.array(six_pip_sound_start_times)

np.save(join(events_folder, 'six_pip_sound_start_times.npy'), six_pip_sound_start_times)
dd.connect_repl_var(globals(), 'six_pip_sound_start_times', 'step_back', 'sync_start', 'args')
# -------------------------

# Find the corresponding sounds in the wav files
# -------------------------
index_of_last_sound_of_rec1 = 2083
sound_to_sync_sampling_ratio = sound_sampling_freq / sync_sampling_freq
six_pip_sound_start_times = np.load(join(events_folder, 'six_pip_sound_start_times.npy'))
sync_first_point = 2168620
rec1_sound_first_point = 1150830
offset =


def sync_start_to_sound_start(sync_start):
    return int(sync_start * sound_to_sync_sampling_ratio)


def sync_step_to_sound_step(sync_step):
    return int(sync_step * sound_to_sync_sampling_ratio)


sync_start = 2160000
sync_step = 60000

sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync_diff')


sound_start = 1300000
sound_step = 274000

tr.connect_repl_var(globals(), 'sync_start', 'sync_start_to_sound_start', 'sound_start')
tr.connect_repl_var(globals(), 'sync_step', 'sync_step_to_sound_step', 'sound_step')

sv.graph_range(globals(), 'sound_start', 'sound_step', 'rec1_sound')


