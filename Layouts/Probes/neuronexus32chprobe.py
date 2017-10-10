import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools


def create_32channels_nn_prb(filename=None, bad_channels=None):
    """
    This function produces a grid with the electrodes positions of
    Neuronexus A1x32-Poly3-5mm-25s-177-CM32 silicone probe
    Parameters
    ----------
    filename -- the filename (with path) of the .prb file for klustakwik
    bad_channels -- a list or numpy array with channels you may want to disregard
    Returns
    -------
    all_electrodes --
    channel_positions -- a Pandas Series with the electrodes positions (in
    two dimensions)
    """

    electrode_amplifier_index_on_grid = np.array([-1, 0, -1,
                                                  -1, -1, -1,
                                                  -1, 31, -1,
                                                  24, -1, 7,
                                                  -1, 1, -1,
                                                  21, -1, 10,
                                                  -1, 30, -1,
                                                  25, -1, 6,
                                                  -1, 15, -1,
                                                  20, -1, 11,
                                                  -1, 16, -1,
                                                  26, -1, 5,
                                                  -1, 14, -1,
                                                  19, -1, 12,
                                                  -1, 17, -1,
                                                  27, -1, 4,
                                                  -1, 8, -1,
                                                  18, -1, 13,
                                                  -1, 23, -1,
                                                  28, -1, 3,
                                                  -1, 9, -1,
                                                  29, -1, 2,
                                                  -1, 22, -1])


    all_electrodes = electrode_amplifier_index_on_grid.reshape((23, 3))
    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes, channel_number=32,
                                  steps_r=3, steps_c=3)


    electrode_coordinate_grid = list(itertools.product(np.arange(1, 24), np.arange(1, 4)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_name_on_grid = np.array(["Int" + str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index=channel_position_indices)
    channel_positions.columns = ['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]

    return all_electrodes, channel_positions