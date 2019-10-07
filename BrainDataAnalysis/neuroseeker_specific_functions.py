

import numpy as np
from scipy import signal
from os.path import join
import re
import pandas as pd

number_of_channels_in_binary_file = 1440
electrode_pitch = 22.5


sampling_freq = 20000


references = np.empty(12*8)
for region in np.arange(12):
    references[region*8:region*8+8] = np.arange(region*120+56, region*120+64, 1)


all_channels_height_on_probe = []
for channel in np.arange(1440):
    row = int(channel / 8) * 2
    if ((channel % 8) + 1) % 2 != 1:
        row += 1
    all_channels_height_on_probe.append(row * electrode_pitch)

all_channels_height = np.array(all_channels_height_on_probe)


def load_binary_amplifier_data(file, number_of_channels=1440, dtype=np.int16):
    raw_extracellular_data = np.memmap(file, mode='r', dtype=dtype)
    raw_extracellular_data = np.reshape(raw_extracellular_data,
                                        (number_of_channels,
                                         int(raw_extracellular_data.shape[0] / number_of_channels)),
                                        order='F')

    return raw_extracellular_data


def load_imfs(file, number_of_channels=72, number_of_imfs=13, dtype=np.int16):
    imfs = np.memmap(file, dtype=dtype, mode='r')
    number_of_timepoints = int(imfs.shape[0] / (number_of_channels * number_of_imfs))
    imfs = np.memmap(file, dtype=dtype, mode='r', shape=(number_of_channels, number_of_imfs, number_of_timepoints))

    return imfs


def load_sync_binary_data(data_folder):
    sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
    sync -= sync.min()


def camel_to_snake_converter(string):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_events_dataframes(basic_folder, event_types):
    event_dataframes = {}
    for event_type in event_types:
        event_dataframes['ev_' + camel_to_snake_converter(event_type)] = \
            pd.read_pickle(join(basic_folder, event_type+".pkl"))
    return event_dataframes


def get_channels_heights_for_spread_calulation(channels_used):
    channels_height_on_probe = []

    for channel in channels_used:
        row = int(channel / 8) * 2
        if ((channel % 8) + 1) % 2 != 1:
            row += 1
        channels_height_on_probe.append(row * electrode_pitch)

    channels_height = np.array(all_channels_height_on_probe)

    return channels_height


def spread_data(data,  channels_height, channels_used=None, row_spacing=0.5):
    if channels_used is None:
        channels_used = np.arange(data.shape[0])

    data_baseline = (data.T - np.median(data, axis=1).T).T

    num_of_rows = len(channels_used)
    new_data = np.zeros(shape=(num_of_rows, np.shape(data)[1]))

    stdv = np.average(np.std(data_baseline, axis=1), axis=0)

    row_spacing = row_spacing * stdv
    for r in np.arange(0, num_of_rows):
            new_data[r, :] = data_baseline[channels_used[r], :] - \
                             row_spacing * 0.5 * channels_height[channels_used[r]] - \
                             r * row_spacing
    return new_data


def downsample(filename, data, factor, filter_type='iir', filter_order=8, zero_phase=True):

    type = data.dtype

    first_channel_downsampled = signal.decimate(data[0], q=factor, n=filter_order, ftype='iir',
                                                      zero_phase=zero_phase)
    first_channel_downsampled = np.squeeze(first_channel_downsampled)
    downsampled_shape = (data.shape[0], len(first_channel_downsampled))

    downsampled_data = np.memmap(filename, dtype=type, mode='w+', shape=downsampled_shape, order='F')

    downsampled_data[0, :] = first_channel_downsampled

    for channel in np.arange(1, downsampled_shape[0]):

        downsampled_data[channel, :] = signal.decimate(data[channel], q=factor, n=filter_order, ftype=filter_type,
                                                             zero_phase=zero_phase)

    return downsampled_data