
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
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

    if prb_filename is not None:
        prb_gen.generate_prb_file(filename=prb_filename, all_electrodes_array=all_electrodes)

    electrode_coordinate_grid = [i for i in list(zip(x_coords, y_coords))]

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



def generate_prb_file(filename, all_electrodes_array, x_coords, y_coords):

    good_channels = [x for x in np.squeeze(np.reshape(all_electrodes_array, (np.size(all_electrodes_array), 1))) if x != -1]

    file = open(filename, 'w')
    file.write('channel_groups = {\n')
    file.write('    # Shank index.\n')
    file.write('    0:\n')
    file.write('        {\n')
    file.write('            # List of channels to keep for spike detection.\n')
    file.write('            \'channels\':   [{},\n'.format(good_channels[0]))
    for channel in good_channels[1:-1]:
        file.write('                           {},\n'.format(channel))
    file.write('                           {}],\n'.format(good_channels[-1]))
    file.write('\n')

    file.write('            # 2D positions of the channels, only for visualization purposes.\n')
    file.write('            # The unit doesn\'t matter.\n')
    file.write('            \'geometry\': {\n')
    for i in range(all_electrodes_array.size):
            electrode = all_electrodes_array[i]
            x = x_coords[i]
            y = y_coords[i]
            if electrode != -1:
                file.write('                {}: ({}, {}),\n'.format(electrode, x, y))
    file.write('            }\n')
    file.write('    }\n')
    file.write('}\n')
    file.close()


def create_1368channels_neuroseeker_prb(base_info_folder, prb_filename):
    lfps_original_indexing = np.array([9, 29, 49, 69, 89, 109, 129, 149, 169, 189, 209,
                                       229, 249, 269, 289, 309, 329, 349, 369, 389, 409, 429,
                                       449, 469, 489, 509, 529, 549, 569, 589, 609, 629, 649,
                                       669, 689, 709, 729, 749, 769, 789, 809, 829, 849, 869,
                                       889, 909, 929, 949, 969, 989, 1009, 1029, 1049, 1069, 1089,
                                       1109, 1129, 1149, 1169, 1189, 1209, 1229, 1249, 1269, 1289, 1309,
                                       1329, 1349, 1369, 1389, 1409, 1429])

    x_coords = np.squeeze(np.load(join(base_info_folder, 'neuroseeker_channel_coordinates_x.npy')))
    y_coords = np.squeeze(np.load(join(base_info_folder, 'neuroseeker_channel_coordinates_y.npy')))

    electrodes = np.arange(1440)

    electrodes_removed = 0

    only_ap_electrodes = []
    only_ap_x_coords = []
    only_ap_y_coords = []
    for old_index in electrodes:
        if not np.in1d(old_index, lfps_original_indexing):
            only_ap_electrodes.append(old_index - electrodes_removed)
            only_ap_x_coords.append(x_coords[old_index])
            only_ap_y_coords.append(y_coords[old_index])
        else:
            electrodes_removed += 1

    only_ap_electrodes = np.array(only_ap_electrodes)
    only_ap_x_coords = np.array(only_ap_x_coords)
    only_ap_y_coords = np.array(only_ap_y_coords)

    generate_prb_file(prb_filename, only_ap_electrodes, only_ap_x_coords, only_ap_y_coords)






