import numpy as np
from os.path import join
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs

import sequence_viewer as seq_v

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_11_AK_40p3 import constants as const



# ----------------------------------------------------------------

# FOLDERS NAMES --------------------------------------------------
experiment = 2

data_folder = join(const.base_save_folder, const.experiment_folders[experiment], 'Data')

binary_data_filename = join(data_folder, 'Amplifier_APs.bin')

kilosort_folder = join(const.base_save_folder, const.experiment_folders[experiment],
                       'Analysis', 'Kilosort')
spyking_circus_folder = join(const.base_save_folder, const.experiment_folders[experiment],
                       'Analysis', r'SpykingCircus\data\data.GUI')

tsne_folder = join(const.base_save_folder, const.experiment_folders[experiment],
                   'Analysis', 'Tsne')

raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)
# ----------------------------------------------------------------



data_cor = raw_data[1050:, :]
data_hyp = raw_data[850:1050, :]
data_th = raw_data[370:850, :]
data_sth = raw_data[:370, :]

buffer = 500
pointer = 20000


def space_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (100*i) for i in np.arange(dat.shape[0])]))
    return result

seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_cor', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_hyp', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_th', transform_name='space_data')

seq_v.graph_range(globals(), 'pointer', 'buffer', 'raw_data', transform_name='space_data')