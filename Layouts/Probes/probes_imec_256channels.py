
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools

def create_256channels_imec_prb(filename=None, steps_r=2, steps_c=2, bad_channels=None):
    r1 = np.array([163, 232, 236, 240, 249, 199, 193, 220, 36, 61, 1, 8, 12, 47, 41])
    r2 = np.array([162, 212, 206, 244, 202, 198, 229, 222, 29, 25, 2, 5, 16, 46, 42])
    r3 = np.array([217, 211, 207, 245, 251, 255, 228, 225, 35, 60, 55, 9, 17, 20, 95])
    r4 = np.array([161, 233, 237, 241, 203, 197, 192, 223, 37, 58, 54, 6, 51, 45, 40])
    r5 = np.array([216, 234, 205, 242, 201, 195, 218, 224, 28, 24, 3, 13, 50, 21, 39])
    r6 = np.array([215, 210, 204, 246, 252, 231, 227, 32, 63, 59, 53, 14, 18, 22, 94])
    r7 = np.array([160, 235, 238, 247, 200, 196, 219, 33, 27, 57, 7, 10, 48, 44, 38])
    r8 = np.array([213, 209, 239, 248, 253, 194, 221, 30, 26, 0, 4, 11, 49, 43, 93])
    r9 = np.array([168, 159, 154, 181, 145, 140, 191, 131, 125, 120, 67, 111, 106, 77, 97])
    r10 = np.array([172, 175, 153, 182, 185, 139, 167, 164, 90, 119, 114, 71, 105, 100, 81])
    r11 = np.array([169, 157, 179, 148, 143, 189, 134, 129, 123, 65, 68, 109, 75, 78, 84])
    r12 = np.array([173, 156, 151, 183, 142, 137, 166, 128, 122, 117, 69, 108, 103, 79, 85])
    r13 = np.array([170, 176, 152, 147, 186, 138, 133, 127, 89, 118, 113, 72, 104, 99, 82])
    r14 = np.array([171, 177, 180, 146, 187, 190, 132, 126, 121, 66, 112, 107, 76, 98, 86])
    r15 = np.array([174, 155, 150, 184, 141, 136, 165, 91, 88, 116, 70, 73, 102, 80, 83])
    r16 = np.array([158, 178, 149, 144, 188, 135, 130, 124, 64, 115, 110, 74, 101, 96, 87])
    r17 = np.array([214, 208, 243, 250, 254, 230, 226, 34, 62, 56, 52, 15, 19, 23, 92])


    all_electrodes_concat = np.concatenate((r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17))
    all_electrodes = all_electrodes_concat.reshape((17, 15))


    #all_electrodes = np.flipud(all_electrodes.T)

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 18), np.arange(1, 16)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = all_electrodes_concat
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))

    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index=channel_position_indices)
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
