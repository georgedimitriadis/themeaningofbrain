

import numpy as np
from os.path import join
import pandas as pd

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.Co4.Decimated_to_Neuroseeker_Density\
    import constants as const

from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "FOLDERS NAMES"

binary_data_filename = join(const.base_save_folder, const.cell_folder, 'Data', const.data_filename)
analysis_folder = join(const.base_save_folder, const.cell_folder, 'Analysis')

decimated_data_folder = join(analysis_folder, const.decimation_type_folder)
kilosort_folder = join(analysis_folder, const.decimation_type_folder, 'Kilosort')

decimated_data_filename = join(decimated_data_folder, r'decimated_data_256channels.bin')

tsne_folder = join(analysis_folder, const.decimation_type_folder, 'TSNE')
tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'


sampling_freq = const.SAMPLING_FREQUENCY

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = Create the ned data set

group_channels = const.group_channels

raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename, const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                                    const.BINARY_FILE_ENCODING)


chunk_size = 36168
number_of_chunks = raw_data.shape[1] / chunk_size

decimated_data = np.memmap(decimated_data_filename, const.BINARY_FILE_ENCODING, mode='w+',
                           shape=raw_data.shape, order='F')


for c in np.arange(number_of_chunks):
    start = int(c * chunk_size)
    end = int(start + chunk_size)
    data_chunk = raw_data[:, start:end]

    for i in np.arange(group_channels.shape[0]):
        decimated_data[group_channels[i], start:end] = np.mean(data_chunk[group_channels[i], :], axis=0)

    if c % 10 == 0:
        print(c)


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

