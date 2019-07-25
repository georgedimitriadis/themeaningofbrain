

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd

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
events_dataframe_file = join(events_folder, 'events.df')

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
sync2 = np.fromfile(rec2_sync_filename, dtype=np.uint16).astype(np.int32)
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


# Use this to clean any mistakenly found starting points
# -------------------------


def step_back(i, step):
    return i-step


s = 1000
args = [s]

sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync_diff')
dd.connect_repl_var(globals(), 'sound_starting_times', 'sync_start', 'step_back', 'args')
# -------------------------
'''

# Find the end points of the 6 pip sounds
# -------------------------
'''
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
dd.connect_repl_var(globals(), 'six_pip_sound_start_times', 'sync_start', 'step_back', 'args')
'''
# -------------------------

# FIND THE CORRESPONDING SOUNDS IN THE WAV FILES
# -------------------------

# Show the sync and wav files together

six_pip_sound_start_times = np.load(join(events_folder, 'six_pip_sound_start_times.npy'))
index_of_last_sound_of_rec1 = 2083
index_of_last_sound_of_rec2 = 2755
# six_pip_sound_start_times = six_pip_sound_start_times[:index_of_last_sound_of_rec2]

sound_to_sync_sampling_ratio = 15
# sync_to_sound_offset = 1150820 - 2168624 * sound_to_sync_sampling_ratio
sync_to_sound_offset = 4833450 - 44259728 * sound_to_sync_sampling_ratio
offset = sync_to_sound_offset

sync_start = 44259728  # For sync 1 2168624
sync_step = 120000

sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync_diff')
# sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync1')
# sv.graph_range(globals(), 'sync_start', 'sync_step', 'sync2')


sound_start = 4833450  # for sync 1 1150820
sound_step = int(sound_to_sync_sampling_ratio*sync_step)
# sv.graph_range(globals(), 'sound_start', 'sound_step', 'rec1_sound')
sv.graph_range(globals(), 'sound_start', 'sound_step', 'rec2_sound')



def sync_start_to_sound_start(sync_start):
    sound_start = int(sync_start * sound_to_sync_sampling_ratio + offset)
    if sound_start < 0:
        sound_start = 0
    return sound_start


def sync_step_to_sound_step(sync_step):
    return int(sync_step * sound_to_sync_sampling_ratio)


tr.connect_repl_var(globals(), 'sync_start', 'sound_start', 'sync_start_to_sound_start')
tr.connect_repl_var(globals(), 'sync_step', 'sound_step', 'sync_step_to_sound_step')


# Calculate manually the different offsets
# Get the long stretches of no sound in the sync file

sync_signal_start = 2168624
sync_signal_end = 43394819
baseline = 5
smallest_sound = 6
times_between_pips = 2380
length_of_six_pip_sound = 12000

# test = sync_diff[sync_signal_start:sync_signal_end]
test = sync_diff[sync_signal_start:]
z = np.squeeze(np.argwhere(np.abs(test) < baseline))
con = np.split(z, np.where(np.diff(z) != 1)[0]+1)

long_flat_stretches = []
starts_of_long_flat_stretches = []
for c in con:
    if len(c) > 0.8 * length_of_six_pip_sound:
        c = np.array(c) + sync_signal_start
        long_flat_stretches.append(c)
        starts_of_long_flat_stretches.append(c[0])

starts_of_long_flat_stretches = np.array(starts_of_long_flat_stretches)


def step_back(i, step):
    return i-step


s = 10000
args = [s]

dd.connect_repl_var(globals(), 'starts_of_long_flat_stretches', 'sync_start', 'step_back', 'args')


# Create manually the offsets given the sound stops of recording
sync_to_sound_1_variable_offsets = {0: -2422350,
                                    4731324: -31585000,
                                    6605624: -31810000,
                                    8288624: -32000000,
                                    9643624: -32470000,
                                    9838624: -32680000,
                                    10189121: -33170000,
                                    12657511: -33490000,
                                    13659363: -33720000,
                                    14562735: -34030000,
                                    15567615: -34250000,
                                    18147333: -34470000,
                                    22867352: -34895000,
                                    23758244: -35240000,
                                    25328134: -35455000,
                                    26901465: -35940000,
                                    28928312: -36390000,
                                    29824538: -36630000,
                                    31038402: -36840000,
                                    32069660: -37070000,
                                    32287437: -37290000,
                                    33076878: -37610000,
                                    33509596: -37840000,
                                    33979573: -38240000,
                                    34761283: -38560000,
                                    38561239: -38560000,
                                    42897970: -39491000}
sync1_starts = np.array(list(sync_to_sound_1_variable_offsets.keys()))
offsets1 = np.array(list(sync_to_sound_1_variable_offsets.values()))

np.save(join(events_folder, 'sync_starts_to_sound1_offsets.npy'), (sync1_starts, offsets1))


sync_to_sound_2_variable_offsets = {44259728: -659062470,
                                    45071505: -659293000,
                                    46629729: -659673000,
                                    47104200: -659903000,
                                    47575608: -659894000,
                                    47975885: -660400000,
                                    48860264: -660880000,
                                    49087793: -661192000,
                                    51318133: -661410000,
                                    53802523: -661630000,
                                    54038749: -661862000,
                                    54941215: -662170000,
                                    55272143: -662510000,
                                    55728442: -662920000,
                                    55932170: -663140000,
                                    57213826: -663350000,
                                    57303930: -663610000,
                                    57534067: -663895000,
                                    57647619: -664320000,
                                    58015924: -664633000,
                                    58104556: -664970000,
                                    58206706: -665440000}
sync2_starts = np.array(list(sync_to_sound_2_variable_offsets.keys()))
offsets2 = np.array(list(sync_to_sound_2_variable_offsets.values()))

np.save(join(events_folder, 'sync_starts_to_sound2_offsets.npy'), (sync2_starts, offsets2))


def get_offset(sync_start):
    try:
        index = np.squeeze(np.argwhere(sync2_starts < sync_start)[-1])
        offset = offsets2[index]
    except:
        offset = sync_to_sound_offset
    return offset


def step_back(i, step):
    return i-step


s = 1000
args = [s]

tr.connect_repl_var(globals(), 'sync_start', 'offset', 'get_offset')
dd.connect_repl_var(globals(), 'six_pip_sound_start_times', 'sync_start', 'step_back', 'args')
# -------------------------


# -------------------------
# FIND FREQUENCY OF SOUNDS
# -------------------------

# Check what is happening and manually correct error

sound_to_sync_sampling_ratio = 15
sync_starts1, sound_1_offsets = np.load(join(events_folder, 'sync_starts_to_sound1_offsets.npy'))
sync_starts2, sound_2_offsets = np.load(join(events_folder, 'sync_starts_to_sound2_offsets.npy'))


def do_nothing(a):
    return a


def sync_to_sound_1(sync_time_point):
    try:
        index = np.squeeze(np.argwhere(sync_starts1 < sync_time_point)[-1])
        offset = sound_1_offsets[index]
        result = int(sync_time_point * sound_to_sync_sampling_ratio + offset)
    except:
        result = 0
    return result


def sync_to_sound_2(sync_time_point):
    try:
        index = np.squeeze(np.argwhere(sync_starts2 < sync_time_point)[-1])
        offset = sound_2_offsets[index]
        result = int(sync_time_point * sound_to_sync_sampling_ratio + offset)
    except:
        result = 0
    return result


def get_sound1_data(start, end):
    return rec1_sound[start:start+end]


def get_sound2_data(start, end):
    return rec2_sound[start:start+end]


def get_freq_and_power(sound_data, figure=None):
    sp = np.fft.fft(sound_data)
    freq = np.fft.fftfreq(len(sound_data)) * sound_sampling_freq
    l = len(freq)
    s = sp[:int(l/2)].real
    f = freq[:int(l/2)]
    p = s.max()
    fm = int(f[np.argwhere(s == p)[0]][0])
    fm = np.around(fm/1000, decimals=0)
    p = int(p)/1000000
    data_peak = sound_data.max()
    if figure is not None:
        figure.clear()
        a = figure.add_subplot(111)
        a.plot(f, s)
    return fm, data_peak


# Manually check if the data are aligned and if the freq calculation works
sync_starting_point = 0
sound_starting_point = 0
dd.connect_repl_var(globals(), 'six_pip_sound_start_times', 'sync_starting_point')
# tr.connect_repl_var(globals(), 'sync_starting_point', 'sound_starting_point', 'sync_to_sound_1')
tr.connect_repl_var(globals(), 'sync_starting_point', 'sound_starting_point', 'sync_to_sound_2')

range = 10000
# sv.graph_range(globals(), 'sync_starting_point', 'range', 'sync1')
sv.graph_range(globals(), 'sync_starting_point', 'range', 'sync_diff')

range_sound = 200000
sv.graph_range(globals(), 'sound_starting_point', 'range_sound', 'rec2_sound')

sound_data = []
args_end_of_data = [150000]
# tr.connect_repl_var(globals(), 'sound_starting_point', 'sound_data', 'get_sound1_data', 'args_end_of_data')
tr.connect_repl_var(globals(), 'sound_starting_point', 'sound_data', 'get_sound2_data', 'args_end_of_data')
osv.graph(globals(), 'sound_data')

freq_and_power = 0
fig = plt.figure(1)
args_spectrum_figure = [fig]
tr.connect_repl_var(globals(), 'sound_data',  'freq_and_power', 'get_freq_and_power', 'args_spectrum_figure')

# this is just to live display the freq and intensity
tr.connect_repl_var(globals(), 'freq_and_power', 'freq_and_power')


# Get the freqs and the intensities at that freq for all sound that are found clean in the sync file
# -------------------------
six_pip_sound_start_times = np.load(join(events_folder, 'six_pip_sound_start_times.npy'))
sound_to_sync_sampling_ratio = 15
sync_starts1, sound_1_offsets = np.load(join(events_folder, 'sync_starts_to_sound1_offsets.npy'))
sync_starts2, sound_2_offsets = np.load(join(events_folder, 'sync_starts_to_sound2_offsets.npy'))
index_of_last_sound_of_rec1 = 2057
index_of_last_sound_of_rec2 = 2754

sound_frequencies = []
sound_intensities = []
for sync_time_point_index in np.arange(len(six_pip_sound_start_times)):
    sync_time_point = six_pip_sound_start_times[sync_time_point_index]
    sync_to_sound = sync_to_sound_1
    get_sound_data = get_sound1_data
    if sync_time_point_index > index_of_last_sound_of_rec1:
        sync_to_sound = sync_to_sound_2
        get_sound_data = get_sound2_data
    sound_time_point = sync_to_sound(sync_time_point)
    data = get_sound_data(sound_time_point, 150000)
    freq, intens = get_freq_and_power(data)
    sound_frequencies.append(freq)
    sound_intensities.append(intens)

events = pd.DataFrame(np.array([six_pip_sound_start_times, sound_frequencies, sound_intensities]).transpose(),
                      columns=['time_points', 'frequencies', 'intensities'])

events.to_pickle(events_dataframe_file)

# Find the muscimol injection points
muscimol_injection_times = np.argwhere(sync_diff > 400)
np.save(join(events_folder, 'muscimol_injection_times.npy'), muscimol_injection_times)
# -------------------------


# Make sure nothing is shifted and find proper intensities
# Make an events dataframe with all the info for each sound (freq, measure intensity, nominal intensity
# -------------------------
events = pd.read_pickle(events_dataframe_file)

freq = 7
ev_7 = events[events['frequencies'] == freq]
plt.hist(ev_7['intensities'])


def look_at_hists_of_intensities(freq, figure):
    ev = events[events['frequencies'] == freq]
    figure.clear()
    a = figure.add_subplot(111)
    a.hist(ev['intensities'], bins=200)


freq = 5
fig = plt.figure(1)
args = [fig]
out = None
sl.connect_repl_var(globals(), 'freq', 'out', 'look_at_hists_of_intensities', 'args', slider_limits=[5, 15])


nominal_intensities_cutoffs = {5: [100, 3690, 6000, 8600, 100000],
                               6: [100, 4700, 8000, 12000, 100000],
                               7: [100, 7500, 12500, 20000, 100000],
                               8: [100, 5000, 10000, 17500, 100000],
                               9: [100, 10000, 15000, 25000, 100000],
                               10: [100, 10000, 20000, 30000, 100000],
                               11: [100, 10000, 20000, 30000, 100000],
                               12: [100, 8000, 15000, 25000, 100000],
                               13: [100, 8000, 15000, 25000, 100000],
                               14: [100, 8000, 15000, 25000, 100000],
                               15: [100, 5000, 12500, 17000, 100000]}
np.save(join(events_folder, 'nominal_intensities_cutoffs.npy'), nominal_intensities_cutoffs)


events = events.assign(nominal_intensities=np.zeros(len(events)))
for freq in nominal_intensities_cutoffs:
    nominal_intensities = pd.cut(events['intensities'].loc[events['frequencies'] == freq].values,
                                 nominal_intensities_cutoffs[freq], labels=False)
    events['nominal_intensities'].loc[events['frequencies'] == freq] = nominal_intensities

events.to_pickle(events_dataframe_file)
# -------------------------

# HAVE A LOOK AT HOW MANY TRIALS I GOT AFTER MUSCIMOL
t = np.argwhere(six_pip_sound_start_times > muscimol_injection_times[-1])
len(t)
len(six_pip_sound_start_times[:index_of_last_sound_of_rec1])

# I GOT 2057 SOUNDS BEFORE THE MUSCIMOL (NOT COUNTING THE FEW I HAVE JUST BEFORE IN THE 2ND RECORDING) AND 383 SOUNDS
# AFTER.

for freq in nominal_intensities_cutoffs:
 print('{}KHz before = {}'. format(str(freq), str(len(events[events['time_points'] <
                                                             six_pip_sound_start_times[index_of_last_sound_of_rec1]]
                                          [events['frequencies'] == freq]))))
 print('{}KHz after = {}'.format(str(freq), str(len(events[events['time_points'] >
                                                           np.squeeze(muscimol_injection_times[-1])]
                                         [events['frequencies'] == freq]))))

# I GOT JUST OVER 30 TRIALS PER FREQUENCY AFTER MUSCIMOL
'''
5KHz before = 191
5KHz after = 34
6KHz before = 188
6KHz after = 34
7KHz before = 193
7KHz after = 36
8KHz before = 188
8KHz after = 38
9KHz before = 178
9KHz after = 39
10KHz before = 192
10KHz after = 34
11KHz before = 193
11KHz after = 35
12KHz before = 191
12KHz after = 33
13KHz before = 180
13KHz after = 37
14KHz before = 188
14KHz after = 34
15KHz before = 171
15KHz after = 29
'''
