

import numpy as np


number_of_channels_in_binary_file = 1440


def load_binary_amplifier_data(file):
    raw_extracellular_data = np.memmap(file, mode='r', dtype=np.int16)
    raw_extracellular_data = np.reshape(raw_extracellular_data,
                                        (number_of_channels_in_binary_file,
                                        int(raw_extracellular_data.shape[0] / number_of_channels_in_binary_file)),
                                        order='F')

    return raw_extracellular_data