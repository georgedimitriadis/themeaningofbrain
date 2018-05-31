__author__ = 'George Dimitriadis'


import numpy as np
from os.path import join
from IO import ephys as ioep
from IO import klustakwik as iokl
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.t_sne_spikes as tsne_spikes
import BrainDataAnalysis.ploting_functions as pf


base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_32ch'

dates = {1: '2014-03-26', 2: '2014-10-17', 3: '2014-11-25'}


all_cell_capture_times = {1: {'2': '05_11_53', '2_1': '05_28_47'},
                          2: {'1': '16_46_02'},
                          3: {'1': '21_27_13', '2': '22_44_57'}}


good_cells_large = {1: ['2'],
                    2: ['1'],
                    3: ['1', '2']}

rat = 1
cell = '2'
good_cells = good_cells_large[rat]


date = dates[rat]
data_folder = join(base_folder + '\\' + date, 'Data')
analysis_folder = join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = all_cell_capture_times[rat]


adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.01
amp_gain = 100
num_ivm_channels = 32
amp_dtype = np.uint16



# 1) Generate the .dat file
filename_raw_data = join(data_folder,
                         'amplifier{}T{}.bin'.format(dates[rat], all_cell_capture_times[rat][cell]))
raw_data = ioep.load_raw_data(filename=filename_raw_data, numchannels=num_ivm_channels, dtype=amp_dtype)
filename_kl_data = join(analysis_folder, r'klustakwik_cell{}\raw_data_klusta.dat'.format(cell))
iokl.make_dat_file(raw_data=raw_data.dataMatrix, num_channels=num_ivm_channels, filename=filename_kl_data)



# Run t-sne
kwx_file_path = join(analysis_folder, 'klustakwik_cell{}'.format(cell),
                     r'threshold_6_5std/threshold_6_5std.kwx')
perplexity = 100
theta = 0.2
iterations = 2000
gpu_mem = 0.8
eta = 200
early_exaggeration = 4.0
indices_of_spikes_to_tsne = None#range(spikes_to_do)
seed = 100000
verbose = 2
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, seed=seed, eta=eta, early_exaggeration=early_exaggeration,
                                indices_of_spikes_to_tsne=indices_of_spikes_to_tsne, verbose=verbose)


# Load t-sne
filename = 't_sne_results_100per_200lr_02theta_2000its_100kseed.npy'
tsne = np.load(join(analysis_folder, 'klustakwik_cell{}'.format(cell), 'threshold_6_5std', filename))


fig, ax = pf.plot_tsne(tsne[:, :seed], color='b')
pf.plot_tsne(tsne[:, seed:(5*seed)], color='g', axes=ax)