
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools
from os.path import join

def create_1440channels_neuroseeker_prb(base_info_folder, connected_channels_filename, prb_filename=None):

    x_coords = np.squeeze(np.load(join(base_info_folder, 'neuroseeker_channel_coordinates_x.npy')))
    y_coords = np.squeeze(np.load(join(base_info_folder, 'neuroseeker_channel_coordinates_y.npy')))
    connected = np.squeeze(np.load(join(base_info_folder, connected_channels_filename)))
    bad_channels = np.squeeze(np.argwhere(connected == False).astype(np.int))

    r1 = np.array([[i, i+1] for i in np.arange(0, 1433, 8)])
    r1 = r1.reshape(r1.size)
    r2 = np.array([[i, i + 1] for i in np.arange(2, 1435, 8)])
    r2 = r2.reshape(r1.size)
    r3 = np.array([[i, i + 1] for i in np.arange(4, 1437, 8)])
    r3 = r3.reshape(r1.size)
    r4 = np.array([[i, i + 1] for i in np.arange(6, 1439, 8)])
    r4 = r4.reshape(r1.size)

    all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
    all_electrodes = all_electrodes_concat.reshape((4, 360))
    all_electrodes = np.flipud(all_electrodes.T)


    if prb_filename is not None:
        prb_gen.generate_prb_file(filename=prb_filename, all_electrodes_array=all_electrodes, channel_number=1440)

    electrode_coordinate_grid = [i for i in list(zip(x_coords, y_coords))]

    #electrode_coordinate_grid = list(itertools.product(np.arange(1, 32), np.arange(1, 5)))
    #electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = np.arange(len(electrode_coordinate_grid))
    electrode_amplifier_name_on_grid = np.array(["Int" + str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index=channel_position_indices)
    channel_positions.columns = ['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]

    return all_electrodes, channel_positions