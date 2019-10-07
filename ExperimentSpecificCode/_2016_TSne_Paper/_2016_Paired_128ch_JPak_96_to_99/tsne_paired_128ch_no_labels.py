__author__ = 'George Dimitriadis'


import numpy as np
from os.path import join
from IO import ephys as ioep
from IO import klustakwik as iokl
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.t_sne_spikes as tsne_spikes
import BrainDataAnalysis.Graphics.ploting_functions as pf


base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch'

dates = {99: '2015-09-09', 98: '2015-09-04', 97: '2015-09-03', 96: '2015-08-28', 94: '2015-08-25', 93: '2015-08-21'}


all_cell_capture_times = {99: {'1': '14_48_07', '2': '15_17_07', '3': '16_03_31', '4': '16_30_44',
                               '5': '17_33_01', '6': '17_46_43', '7': '18_21_44', '7_1': '18_38_50',
                               '8': '18_58_02', '9': '19_14_01', '10': '19_40_00', '11': '20_26_29',
                               '12': '20_37_00', '13': '20_43_45'},
                          98: {'1': '15_42_02', '2': '15_48_15', '3': '16_14_37', '4': '16_52_15',
                               '4_1': '17_10_23', '5': '18_16_25', '6': '19_08_39', '6_1': '19_24_34',
                               '7': '20_12_32'},
                          97: {'1': '15_32_01', '2': '16_21_00', '2_1': '16_37_50', '3': '17_18_47',
                               '3_1': '17_40_01', '4': '18_13_52', '5': '18_47_20', '6': '19_01_02',
                               '6_1': '19_40_01', '7': '20_11_01', '8': '20_50_17', '9': '21_18_47'},
                          96: {'1': '18_50_42', '2': '19_03_02', '2_1': '19_48_15', '2_2': '20_15_45',
                               '3': '21_04_38', '4': '21_35_13', '5': '22_09_44', '6': '22_43_37',
                               '7': '22_58_30', '8': '23_09_26', '9': '23_28_20'},
                          94: {'1': '20_31_40', '2': '23_21_12', '3': '00_37_00', '4': '00_56_29'},
                          93: {'1': '22_53_29', '2': '23_23_22', '3': '23_57_25', '3_1': '00_23_38'}}

all_spike_thresholds = {99: {'1': 1e-3, '2': 1e-3, '3': 1e-3, '4': 1e-3,
                             '5': 3e-4, '6': 1e-3, '7': 1e-3, '7_1': 1e-3,
                             '8': 1e-3, '9': 1e-3, '10': 1e-3, '11': 1e-3,
                             '12': 1e-3, '13': 1e-3},
                        98: {'1': 6e-4, '2': 1e-3, '3': 1e-3, '4': 1e-3,
                             '4_1': 3e-4, '5': -1e-3, '6': 1e-3, '6_1': 1e-3,
                             '7': 1e-3},
                        97: {'1': 1e-3, '2': 1e-3, '2_1': 1e-3, '3': 1e-3,
                             '3_1': 1e-3, '4': 2e-3, '5': 1e-3, '6': 1e-3,
                             '6_1': 2e-3, '7': 5e-4, '8': 1e-3, '9': -3e-4},
                        96: {'1': 1e-3, '2': 1e-3, '2_1': 1e-3, '2_2': 1e-3,
                             '3': 1e-3, '4': 1e-3, '5': 1e-3, '6': 1e-3,
                             '7': 1e-3, '8': 1e-3, '9': 1e-3},
                        94: {'1': 1e-3, '2': 1e-3, '3': 1e-3, '4': -5e-4},
                        93: {'1': 1e-3, '2': 5e-4, '3': 4e-4, '3_1': 4e-4}}

all_throwing_away_spikes_thresholds = {99: {'4': -300, '6': -300, '7': -300, '7_1': -300},
                                       98: {'5': -800},
                                       97: {'6': -1000, '6_1': -1000, '9': -2000},
                                       96: {'2_1': -400, '2_2': -400, '7': -400, '9': -500},
                                       94: {'1': -300, '2': -300, '3': -300, '4': -500},
                                       93: {'1': -300, '2': -300, '3': -300, '3_1': -400}}

all_average_over_n_spikes = {99: {'4': 60, '6': 60, '7': 60, '7_1': 60},
                             98: {'5': 30},
                             97: {'6': 200, '6_1': 200, '9': 10},
                             96: {'2_1': 100, '2_2': 100, '7': 100, '9': 300},
                             94: {'1': 100, '2': 100, '3': 100, '4': 100},
                             93: {'1': 100, '2': 100, '3': 100, '3_1': 50}}

good_cells_joana = {99: ['4', '6', '7', '7_1'],
                    98: ['3', '4_1', '5', '6_1'],
                    97: ['4', '6', '6_1', '7', '9'],
                    96: ['2_1', '2_2', '7', '9'],
                    94: ['4'],
                    93: ['1', '2', '3', '3_1']}

good_cells_large = {97: ['9'],
                    98: ['5'],
                    93: ['3_1']}

rat = 93
good_cells = good_cells_large[rat]


date = dates[rat]
data_folder = join(base_folder + '\\' + date, 'Data')
analysis_folder = join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = all_cell_capture_times[rat]
spike_thresholds = all_spike_thresholds[rat]
throw_away_spikes_thresholds = all_throwing_away_spikes_thresholds[rat]
average_over_n_spikes = all_average_over_n_spikes[rat]


num_of_points_in_spike_trig_ivm = 64
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.01
amp_gain = 100
num_ivm_channels = 128
amp_dtype = np.uint16



# 1) Generate the .dat file
filename_raw_data = join(data_folder,
                         'amplifier{}T{}.bin'.format(dates[rat], all_cell_capture_times[rat][good_cells[0]]))
raw_data = ioep.load_raw_data(filename=filename_raw_data, numchannels=num_ivm_channels, dtype=amp_dtype)
filename_kl_data = join(analysis_folder, r'klustakwik_cell{}\raw_data_klusta.dat'.format(good_cells[0]))
iokl.make_dat_file(raw_data=raw_data.dataMatrix, num_channels=num_ivm_channels, filename=filename_kl_data)



# Run t-sne
kwx_file_path = join(analysis_folder, 'klustakwik_cell{}'.format(good_cells[0]),
                     r'threshold_6_5std/threshold_6_5std.kwx')
perplexity = 100
theta = 0.2
iterations = 2000
gpu_mem = 0.8
eta = 200
early_exaggeration = 4.0
indices_of_spikes_to_tsne=range(34000)
seed = 0
verbose = 2
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, seed=seed, eta=eta, early_exaggeration=early_exaggeration,
                                indices_of_spikes_to_tsne=indices_of_spikes_to_tsne, verbose=verbose)



pf.plot_tsne(tsne)
