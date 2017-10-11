__author__ = 'George Dimitriadis'

import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools

def create_128channels_imec_prb(filename=None, steps_r=2, steps_c=2, bad_channels=None):
    r1 = np.array([103, 101, 99, 97, 95, 93, 91, 89, 87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41, 1, 61, 57, 36,
                   34, 32,	30,	28,	26,	24,	22,	20])
    r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55, 2, 62, 58,
                   4, 6, 8,	10,	12,	14,	21,	19,	16])
    r3 = np.array([102,	100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59, 39,
                   37, 35, 33, 31, 29, 27, 25, 23])
    r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	79,	113, 48, 50, 52, 54, 56, 0, 60,
                   3, 5, 7,	9, 11, 13, 15, 18, -1])


    all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
    all_electrodes = all_electrodes_concat.reshape((4, 32))


    all_electrodes = np.flipud(all_electrodes.T)

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 5), np.arange(1, 33)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = all_electrodes_concat
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))

    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        #channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
        channel_positions[channel_positions.Numbers.isin(bad_channels)] = -1
        bad_channels_mask = np.reshape(np.in1d(all_electrodes, bad_channels), (32, 4))
        min_one_val = np.empty_like(bad_channels)
        min_one_val.fill(-1)
        np.place(all_electrodes, bad_channels_mask, min_one_val)

    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes,
                                  steps_r=steps_r, steps_c=steps_c)

    return all_electrodes, channel_positions


