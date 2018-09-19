


import scipy.signal as ssig
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker Chronic Rat 22.1\2017_05_29\13_30_53\Analysis\Kilosort'
data_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker Chronic Rat 22.1\2017_05_29\13_30_53\Data'
binary_data_filename = join(data_folder, r'2017_05_29T13_30_53_Amp_S16_LP3p5KHz_uV.bin')

probe_info_folder = r'E:\George\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = join(probe_info_folder, 'prb.txt')

time_points = 100
sampling_frequency = 20000

number_of_channels_in_binary_file = 1440



'''
# Call to clean the kilosort generated templates
from GUIs.Kilosort import clean_kilosort_templates as clean

clean.cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename, prb_file,
                              sampling_frequency=20000)
'''





# Generate Square wave pulse train to recapture Camera TTL information
number_of_channels_in_binary_file = 1440

binary_data_filename = join(data_folder, r'2017_05_29T13_30_53_Amp_S16_LP3p5KHz_uV.bin')
pulse_data_trace_filename = r'2017_05_29T13_30_53_Sync_U16.bin'

raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

pulse_data = np.memmap(join(data_folder, pulse_data_trace_filename), dtype=np.uint16, mode='r')


def plot_both_pulses(pulse_data, pulse_square=None, pulse_freq=None, sampling_frequency = 20000, start_time=0, end_time=3600, step_time=1, time_window=0.5):
    plt.interactive(True)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    start_time = start_time
    end_time = end_time
    step_time = step_time
    time_window = time_window
    num_of_windows = int((end_time - start_time) / step_time)

    for win in range(num_of_windows):
        st = start_time + step_time * win
        et = st + time_window
        stp = int(st * sampling_frequency)
        etp = int(et * sampling_frequency)
        t = np.linspace(st, et, etp - stp)
        if pulse_square is None:
            square = ssig.square(2 * np.pi * pulse_freq * (t + 20/sampling_frequency), duty=1.0-0.0434) / 2 + 0.5 + 65278
        else:
            square = pulse_square[stp:etp]

        ax.clear()
        ax.plot(t, square, t, pulse_data[stp:etp])

        plt.waitforbuttonpress()



top_of_pulse_points = np.argwhere(pulse_data==65279)
bottom_of_pulse_points = np.argwhere(pulse_data==65278)
starting_pulse = bottom_of_pulse_points[0][0]-167
end_pulse = top_of_pulse_points[-1][0]
time_points_in_ttl_train = end_pulse - starting_pulse
time_of_frames_train = 5453.015 # 5422.866 # From the csv file
sampling_frequency_corrected = time_points_in_ttl_train / time_of_frames_train



pulse_freq = 119.6057583 # 119.6057583 # 120.27072
full_time = pulse_data.shape[0] / sampling_frequency_corrected
plot_both_pulses(pulse_data, pulse_freq=pulse_freq, sampling_frequency=sampling_frequency_corrected,
                 start_time=15, end_time=full_time, step_time=2)







t = np.linspace(0, full_time, pulse_data.shape[0])

square = ssig.square(2 * np.pi * pulse_freq * (t + 20/sampling_frequency_corrected), duty=1.0-0.0434) / 2 + 0.5 + 65278
square[:starting_pulse] = 65278
square[end_pulse:] = 65278

plot_both_pulses(pulse_data, pulse_square=square, start_time=full_time-60, end_time=full_time+1, step_time=0.5)

stp = int(14 * sampling_frequency_corrected)
etp = int(18 * sampling_frequency_corrected)
plt.plot(t[stp:etp], square[stp:etp], t[stp:etp], pulse_data[stp:etp])

stp = int((full_time - 15) * sampling_frequency_corrected)
etp = int((full_time - 10) * sampling_frequency_corrected)
plt.plot(t[stp:etp], square[stp:etp], t[stp:etp], pulse_data[stp:etp])

np.save(join(data_folder, r'corrected_camera_ttl_pulses.npy'), square)
square = np.load(join(data_folder, r'corrected_camera_ttl_pulses.npy'))




transitions = np.diff(square)
num_of_pulses = np.sum(transitions==-1)






frame_times = np.load(r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker Chronic Rat 22.1\2017_05_29\13_30_53\Data\frame_times.npy')

csv_frames = np.ones(pulse_data.shape[0])*65278
frame_times_offseted = frame_times + 15.9348
for frame_time in frame_times_offseted:
    csv_frames[int(frame_time*sampling_frequency_corrected)] = 65279

stp = int(14 * sampling_frequency_corrected)
etp = int(18 * sampling_frequency_corrected)
plt.plot(t[stp:etp], csv_frames[stp:etp], t[stp:etp], pulse_data[stp:etp])

stp = int((full_time - 45) * sampling_frequency_corrected)
etp = int((full_time - 35) * sampling_frequency_corrected)
plt.plot(t[stp:etp], csv_frames[stp:etp], t[stp:etp], pulse_data[stp:etp])


plot_both_pulses(pulse_data, pulse_square=csv_frames, start_time=0, end_time=full_time+1, step_time=2)