
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, savgol_filter
from copy import copy

global use_dask
use_dask = True
try:
    import dask
except:
    use_dask = False


def load_binary_amplifier_data(file, number_of_channels, type):
    raw_extracellular_data = np.memmap(file, mode='r', dtype=type)
    raw_extracellular_data = np.reshape(raw_extracellular_data,
                                        (number_of_channels,
                                         int(raw_extracellular_data.shape[0] / number_of_channels)),
                                        order='F')

    return raw_extracellular_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def denoise_data(binary_filename, binary_type, result_filename, number_of_channels, sampling_frequency,
                 compute_window_in_secs=1, use_dask_for_parallel=True):

    global number_of_points
    number_of_points = sampling_frequency * compute_window_in_secs

    global raw_data
    raw_data = load_binary_amplifier_data(binary_filename, number_of_channels, binary_type)

    global denoised_data
    denoised_data = np.memmap(result_filename, binary_type, mode='w+', shape=raw_data.shape, order='F')

    pane_index = np.arange(int(raw_data.shape[1] / number_of_points) + 1)

    global number_of_data_windows
    number_of_data_windows = len(pane_index)

    global processes_done
    processes_done = 0

    def create_spike_detection_mask(data, low_cut=1500, high_cut=2500, fs=20000, order=5, channel_group=50):

        filtered_bp = butter_bandpass_filter(data, low_cut, high_cut, fs, order)
        z_f = hilbert(filtered_bp, axis=1)
        phases = np.cos(np.unwrap(np.angle(z_f)))

        d_data = np.concatenate((np.diff(data, axis=1), np.zeros((data.shape[0], 1))), axis=1)
        d_data_smooth = savgol_filter(d_data, 11, 3)

        phases = d_data_smooth * phases
        phases_m_f_g = []

        for c in range(len(data)):
            if c < len(data) - channel_group:
                chans = range(c, c + channel_group)
                phases_m_f_g.append(savgol_filter(np.abs(np.sum(phases[chans, :], axis=0)), 31, 3))
            else:
                phases_m_f_g.append(np.ones(data.shape[1]))

        phases_m_f_g = np.array(phases_m_f_g)

        return phases_m_f_g

    def cut_data(start, end):
        global number_of_points
        global raw_data
        data = copy(raw_data[:, int(start):int(end)])
        return data

    def transform_data(data, mask):
        new_data = mask * data / 100  # Divide by a number (100) to bring the numbers back into int16 range
        new_data = new_data.astype(np.int16)
        return new_data

    def assign(data, start, end):
        global raw_data
        global denoised_data
        denoised_data[:, start:end] = data

        print('        Done {} of {}'.format(str(start / number_of_points), str(int(raw_data.shape[1] / number_of_points) + 1)))

        return end

    def process(i):
        global number_of_points
        global number_of_data_windows
        global processes_done
        print('Start process {} of {}'.format(str(processes_done), str(number_of_data_windows)))
        this_process_id = processes_done
        processes_done += 1

        start = int(i * number_of_points)
        end = int(number_of_points * i + number_of_points)

        data = cut_data(start, end)
        mask = create_spike_detection_mask(data)
        new_data = transform_data(data, mask)
        assign(new_data, start, end)
        print('Finished process {} of {}'.format(str(this_process_id), str(number_of_data_windows)))

    if use_dask and use_dask_for_parallel:
        result = [dask.delayed(process)(i) for i in pane_index]
        dask.compute(*result)
    else:
        [process(i) for i in pane_index]

    #remaining_points = raw_data.shape[1] - pane_index[-1] * number_of_points