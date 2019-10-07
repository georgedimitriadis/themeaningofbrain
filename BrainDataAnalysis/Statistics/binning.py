
import pandas as pd
import numpy as np


def bin_2d_array(data, bins=[10, 10]):

    number_of_bins = [0, 0]
    for i in range(2):
        if bins[i].__class__ is not int:
            number_of_bins[i] = len(bins[i]) - 1
        else:
            number_of_bins[i] = bins[i]

    out_first = pd.cut(data[:, 0], bins=bins[0], labels=np.arange(number_of_bins[0]))

    out_second = pd.cut(data[:, 1], bins=bins[1], labels=np.arange(1000, 1000 * (number_of_bins[1] + 1), 1000))

    out = pd.Categorical(out_first.get_values() + out_second.get_values())

    return out


def spike_count_per_frame(template_info, spike_info, ev_video, sampling_frequency, file_to_save_to=None):
    frame_starts = ev_video['AmpTimePoints'].values
    timepoints_in_frame = np.nanmedian(ev_video['AmpTimePoints'].diff())
    seconds_per_frame = timepoints_in_frame / sampling_frequency

    spike_rates = np.zeros((len(template_info), len(frame_starts)))

    for t_index in np.arange(len(template_info)):
        template_index = template_info['template number'].iloc[t_index]
        spike_times_in_template = spike_info[spike_info['template_after_sorting'] == template_index]['times'].values

        for s_index in np.arange(len(frame_starts)):
            start = frame_starts[s_index]
            try:
                end = start + timepoints_in_frame
                spikes_in_window = spike_times_in_template[
                    np.logical_and(spike_times_in_template < end, spike_times_in_template > start)]
                spike_rates[t_index, s_index] = len(spikes_in_window) / seconds_per_frame
            except:
                pass

        print('Done neuron {} out of {}'.format(str(t_index), str(len(template_info))))

    if file_to_save_to is not None:
        np.save(file_to_save_to, spike_rates)

    return spike_rates


def rolling_window_with_step(data, function, window_size, step):
    """
    Runs the function over windows of window_size with step. Can deal with 1D and 2D data. For 2D data there are two
    assumptions. 1) The function runs over the second axis (axis=1) and 2) The function accepts the axis parameter
    :param data: The data to pass the function over
    :param function: The function to run
    :param window_size: The window size
    :param step: The step
    :return:
    """
    if len(np.shape(data)) == 1:
        data_len = len(data)

        start_indices = np.arange(0, data_len, step).astype(np.int)

        result = [function(data[i:i + int(window_size)])
                  for i in start_indices if i + window_size < data_len]

        i = int(len(result) * window_size)
        if i < data_len - 1:
            result.append(function(data[i: -1]))

    elif len(np.shape(data)) == 2:
            data_len = np.shape(data)[1]

            start_indices = np.arange(0, data_len, step).astype(np.int)

            result = [function(data[:, i:i + int(window_size)], axis=1)
                      for i in start_indices if i + window_size < data_len]

            i = int(len(result) * window_size)
            if i < data_len - 1:
                result.append(function(data[:, i: -1], axis=1))

            result = np.transpose(result)

    return result


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    if len(denom.shape) == 0:
        if denom == 0:
            denom = 1
    else:
        denom[denom==0] = 1
    return x_min + nom/denom