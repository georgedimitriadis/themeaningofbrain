__author__ = 'George Dimitriadis'

import numpy as np
from os.path import dirname, exists, join
from os import makedirs
from shutil import copyfile
from Layouts.Probes import probes_imec as pi
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.t_sne_spikes as tsne_spikes
import BrainDataAnalysis.ploting_functions as pf
import BrainDataAnalysis.tsne_analysis_functions as taf
import matplotlib.pyplot as plt
import h5py as h5
import BrainDataAnalysis.Utilities as ut
import pandas as pd

raw_juxta_data_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Data\adc2015-09-03T21_18_47.bin'
raw_data_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Data\amplifier2015-09-03T21_18_47.bin'
basic_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_variate_probe_geometries'
param_filename = join(basic_dir, 'threshold_6_5std.prm')
geometry_dir = r'{}_channels_{}'
prb_filename = r'{}.prb'

geometry_codes = ['64_1', '64_2', '64_3', '64_4', '64_5', '32_1', '32_2', '32_t', '22_1', '16_1', '16_2', '8_1', '8_2']

geometry_descriptions = {'64_1': 'columns_1_and_2',
                         '64_2': 'columns_1_and_3',
                         '64_3': 'columns_1_and_4',
                         '64_4': 'alternate_rows',
                         '64_5': 'alternate_electrodes',
                         '32_0': 'like_neuronexus',
                         '32_1': 'columns_1_and_3_alternate_rows',
                         '32_2': 'columns_1_and_4_alternate_rows',
                         '32_t': 'tetrodes_spaced_40um_apart',
                         '22_1': 'every_3_electrodes',
                         '16_1': 'columns_1_and_3_every_4_rows',
                         '16_2': 'columns_1_and_4_every_4_rows',
                         '8_1': 'columns_1_and_3_every_8_rows',
                         '8_2': 'columns_1_and_4_every_8_rows'}

channel_number = {'64_1': 64,
                  '64_2': 64,
                  '64_3': 64,
                  '64_4': 64,
                  '64_5': 64,
                  '32_0': 32,
                  '32_1': 32,
                  '32_2': 32,
                  '32_t': 32,
                  '22_1': 22,
                  '16_1': 16,
                  '16_2': 16,
                  '8_1': 8,
                  '8_2': 8}


steps_r = {'64_1': 2,
           '64_2': 2,
           '64_3': 2,
           '64_4': 3,
           '64_5': 2,
           '32_1': 3,
           '32_2': 3,
           '32_t': 3,
           '22_1': 4,
           '16_1': 2,
           '16_2': 3,
           '8_1': 2,
           '8_2': 3}

steps_c = {'64_1': 2,
           '64_2': 3,
           '64_3': 4,
           '64_4': 2,
           '64_5': 2,
           '32_1': 2,
           '32_2': 3,
           '32_t': 3,
           '22_1': 4,
           '16_1': 5,
           '16_2': 5,
           '8_1': 8,
           '8_2': 8}

bad_channels = {}

r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

spike_channels = np.concatenate((r1[3:7], r2[3:7], r3[3:7]))


def ensure_dir(f):
    d = dirname(f)
    if not exists(d):
        makedirs(d)
    return f

# 64 CHANNELS GEOMETRIES
# COLUMNS
# Columns 1&2
r3 = np.array([102,	100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59, 39,
               37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	79,	113, 48, 50, 52, 54, 56, 0, 60,
               3, 5, 7,	9, 11, 13, 15, 18, -1])
bad_channels['64_1'] = np.concatenate((r3, r4))


# Columns 1&3
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55, 2, 62, 58,
               4, 6, 8,	10,	12,	14,	21,	19,	16])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	79,	113, 48, 50, 52, 54, 56, 0, 60,
               3, 5, 7,	9, 11, 13, 15, 18, -1])
bad_channels['64_2'] = np.concatenate((r2, r4))

# Columns 1&4
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55, 2, 62, 58,
               4, 6, 8,	10,	12,	14,	21,	19,	16])
r3 = np.array([102,	100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59, 39,
               37, 35, 33, 31, 29, 27, 25, 23])
bad_channels['64_3'] = np.concatenate((r2, r3))


# ROWS
# Every Other row
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_other = np.arange(1, len(r2)+1, 2)
r1 = r1[every_other]
r2 = r2[every_other]
r3 = r3[every_other]
r4 = r4[every_other]
bad_channels['64_4'] = np.concatenate((r1, r2, r3, r4))

# Every Other electrode
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_other_even_rows = np.arange(1, len(r2)+1, 2)
every_other_odd_rows = np.arange(0, len(r2), 2)
r1 = r1[every_other_odd_rows]
r2 = r2[every_other_even_rows]
r3 = r3[every_other_odd_rows]
r4 = r4[every_other_even_rows]
bad_channels['64_5'] = np.concatenate((r1, r2, r3, r4))


# 32 CHANNELS GEOMETRIES

# Columns 1 and 3 and every 2 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_other = np.arange(1, len(r2)+1, 2)
r1 = r1[every_other]
r3 = r3[every_other]
bad_channels['32_1'] = np.concatenate((r1, r2, r3, r4))

# Columns 1 and 4 and every 4 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_other = np.arange(1, len(r2)+1, 2)
r1 = r1[every_other]
r4 = r4[every_other]
bad_channels['32_2'] = np.concatenate((r1, r2, r3, r4))


# Tetrodes spaced 40um (every three electrodes) apart
r1 = [103, 101, 99, 97, 91, 89, 87, 70, 66, 84, 108, 110, 47, 45, 43, 41, 57, 36, 34, 32, 30, 24, 22, 20]
r2 = [106, 104, 115, 117, 123, 125, 127, 71, 67, 74, 114, 112, 49, 51, 53, 55, 58, 4, 6, 8, 10, 21, 19, 16]
r3 = [98,  96,  94,  92,  90,  88, 68, 65, 85, 83,  81, 111, 42, 40, 38, 63, 59, 39, 33, 31, 29, 27, 25, 23]
r4 = [105, 116, 118, 120, 122, 124, 69,	64,	75,	77,	 79, 113, 52, 54, 56, 0, 60, 3, 9, 11, 13, 15, 18, -1]
bad_channels['32_t'] = np.concatenate((r1, r2, r3, r4))

# 22 CHANNELS GEOMETRIES

# Every 2 electrodes
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_three = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31]
r1 = r1[every_three]
r4 = r4[every_three]
bad_channels['22_1'] = np.concatenate((r1, r2, r3, r4))


# 16 CHANNELS GEOMETRIES

# Columns 1 and 3 and every 4 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_four = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
r1 = r1[every_four]
r3 = r3[every_four]
bad_channels['16_1'] = np.concatenate((r1, r2, r3, r4))

# Columns 1 and 4 and every 4 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_four = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
r1 = r1[every_four]
r4 = r4[every_four]
bad_channels['16_2'] = np.concatenate((r1, r2, r3, r4))


# 8 CHANNELS GEOMETRIES

# Columns 1 and 3 and every 8 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_eight = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31]
r1 = r1[every_eight]
r3 = r3[every_eight]
bad_channels['8_1'] = np.concatenate((r1, r2, r3, r4))

# Columns 1 and 4 and every 8 rows
r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

every_eight = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31]
r1 = r1[every_eight]
r4 = r4[every_eight]
bad_channels['8_2'] = np.concatenate((r1, r2, r3, r4))

# ----------------------------------------------------------------------------------------------------------------------
# CODE TO GENERATE THE .PRM and .PRB FILES
def create_files_for_code(code):
    dir_and_file_name = ensure_dir(join(basic_dir, geometry_dir.format(channel_number[code],
                                        geometry_descriptions[code]),
                                        prb_filename.format(geometry_descriptions[code])))

    all_electrodes, channel_positions = pi.create_128channels_imec_prb(filename=dir_and_file_name,
                                                                       bad_channels=bad_channels[code],
                                                                       steps_r=steps_r[code], steps_c=steps_c[code])

    new_param_filename = join(basic_dir, geometry_dir.format(channel_number[code],
                                                             geometry_descriptions[code]), 'threshold_6_5std.prm')
    copyfile(param_filename, new_param_filename)
    text_in_prm = r"experiment_name = 'threshold_6_5std'" + '\n'
    text_in_prm = text_in_prm + "prb_file = r'" + join(basic_dir, dir_and_file_name) + "'"
    file = open(new_param_filename, 'r+')
    content = file.read()
    file.seek(0, 0)
    file.write(text_in_prm + '\n' + content)
    file.close()

# Do all codes
for code in geometry_codes:
    create_files_for_code(code)

# Do a specific code
code = '32_t'
create_files_for_code(code)


# ----------------------------------------------------------------------------------------------------------------------
# T-SNE
# Run t-sne
code = '32_t'
kwx_file_path = join(basic_dir, geometry_dir.format(channel_number[code], geometry_descriptions[code]),
                     'threshold_6_5std.kwx')
perplexity = 100
theta = 0.2
iterations = 5000
gpu_mem = 0.8
eta = 200
early_exaggeration = 4.0
indices_of_spikes_to_tsne = None#range(spikes_to_do)
seed = 0
verbose = 2
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, seed=seed, eta=eta, early_exaggeration=early_exaggeration,
                                indices_of_spikes_to_tsne=indices_of_spikes_to_tsne, verbose=verbose)



# Load t-sne
def load_tsne(code):
    filename = join(basic_dir, geometry_dir.format(channel_number[code], geometry_descriptions[code]),
                    't_sne_results.npy')
    tsne = np.load(filename)
    return tsne


tsne = load_tsne(code)


# ----------------------------------------------------------------------------------------------------------------------
# JUXTA SPIKES
# Get extra spike times (as found by phy detect)

spike_thresholds = -2e-4
adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.002
amp_gain = 100
num_of_raw_data_channels = 128

kwik_file = join(basic_dir, geometry_dir.format(channel_number[code], geometry_descriptions[code]),
                 'threshold_6_5std.kwik')
juxta_cluster_indices_grouped, spike_thresholds_groups = taf.create_juxta_label(kwik_file,
                                                                                spike_thresholds=spike_thresholds,
                                                                                adc_channel_used=adc_channel_used,
                                                                                adc_dtype=adc_dtype,
                                                                                inter_spike_time_distance=inter_spike_time_distance,
                                                                                amp_gain=amp_gain,
                                                                                num_of_raw_data_channels=None,
                                                                                spike_channels=None,
                                                                                verbose=True)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING

pf.plot_tsne(tsne)


pf.plot_tsne(tsne, juxta_cluster_indices_grouped, cm=plt.cm.brg,
             label_name='Peak size in uV',
             label_array=(spike_thresholds_groups*1e6).astype(int))



# Show all t-snes
for code in geometry_codes:
    print('---'+code+'---')
    tsne = load_tsne(code)
    kwik_file = join(basic_dir, geometry_dir.format(channel_number[code], geometry_descriptions[code]),
                 'threshold_6_5std.kwik')
    juxta_cluster_indices_grouped, spike_thresholds_groups = taf.create_juxta_label(kwik_file,
                                                                                    spike_thresholds=spike_thresholds,
                                                                                    adc_channel_used=adc_channel_used,
                                                                                    adc_dtype=adc_dtype,
                                                                                    inter_spike_time_distance=inter_spike_time_distance,
                                                                                    amp_gain=amp_gain,
                                                                                    num_of_raw_data_channels=None,
                                                                                    spike_channels=None,
                                                                                    verbose=True)
    pf.plot_tsne(tsne, juxta_cluster_indices_grouped, cm=plt.cm.brg,
             label_name='Peak size in uV',
             label_array=(spike_thresholds_groups*1e6).astype(int),
             subtitle='T-sne of '+str(channel_number[code])+'channels, '+str(geometry_descriptions[code]))


# ----------------------------------------------------------------------------------------------------------------------
# CLUSTERING AND SCORING

db, n_clusters, labels, core_samples_mask, score = taf.fit_dbscan(tsne, 0.019, 40, normalize=True, show=True)

pf.show_clustered_tsne(db, tsne, juxta_cluster_indices_grouped=None, threshold_legend=None)


taf.calculate_precision_recall_for_single_label_grouped(tsne, juxta_cluster_indices_grouped, n_clusters, labels,
                                                        core_samples_mask, show_means=False)

# ----------------------------------------------------------------------------------------------------------------------
# Comparing tsnes with the manual clustering of the 128 channels

base_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03'
tsne_dir = r'Analysis\klustakwik\threshold_6_5std'
cluster_info_filename = join(base_dir, tsne_dir, 'cluster_info.pkl')
clusters = pd.read_pickle(cluster_info_filename)['Spike_Indices'].tolist()

kwik_file = join(base_dir, tsne_dir, 'threshold_6_5std.kwik')
h5file = h5.File(kwik_file, mode='r')
extra_spike_times_128ch = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()

clusters_subselection_all = {}

code = '32_t'

def gen_clusters_subselection(clusters_subselection_all, code):
    kwik_file = join(basic_dir, geometry_dir.format(channel_number[code], geometry_descriptions[code]),
                     'threshold_6_5std.kwik')
    h5file = h5.File(kwik_file, mode='r')
    extra_spike_times_subselection = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
    h5file.close()

    clusters_subselection = {}
    for c in np.arange(len(clusters)):
        common_spikes, indices_of_common_spikes, not_common_spikes = \
            ut.find_points_in_array_with_jitter(extra_spike_times_128ch[clusters[c]], extra_spike_times_subselection, 7)
        clusters_subselection[c] = indices_of_common_spikes

    clusters_subselection_all[code] = clusters_subselection

    return clusters_subselection_all


# Just for the 128 channels case
clusters_subselection = {}
for c in np.arange(len(clusters)):
    clusters_subselection[c] = clusters[c]
#------------------------------


codes = ['32_t']

# use to make a randomizer of the color map used
temp_for_remaping = np.arange(len(clusters))
cm_remaping = {}
for c in np.arange(len(clusters)):
    cm_remaping[c] = np.random.choice(temp_for_remaping[c:])


cm_remaping = np.load(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_variate_probe_geometries\pics\Labeled with jet randomized\cm_remapping.npy')

for code in codes:
    clusters_subselection_all = gen_clusters_subselection(clusters_subselection_all, code)

    tsne = load_tsne(code)
    pf.plot_tsne(tsne, clusters_subselection_all[code], subtitle=code, cm_remapping=cm_remaping, cm=plt.cm.jet, legent_on=False)



