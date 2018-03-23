

import numpy as np


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


def load_binary_amplifier_data(file):
    raw_extracellular_data = np.memmap(file, mode='r', dtype=np.int16)
    raw_extracellular_data = np.reshape(raw_extracellular_data,
                                        (number_of_channels_in_binary_file,
                                         int(raw_extracellular_data.shape[0] / number_of_channels_in_binary_file)),
                                        order='F')

    return raw_extracellular_data


def spread_data(data, channels_height, channels_used, row_spacing=0.5):

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