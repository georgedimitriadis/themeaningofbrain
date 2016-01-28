__author__ = 'George Dimitriadis'


import numpy as np
import pandas as pd
import itertools


def probe_layout_E32B_20_S04_L10_200_single_shaft(shaft='A'):
    electrode_coordinate_grid = [(0,0),(-1,2),(1,4),(-2,6),(2,8),(-3,10),(3,12),(-4,14)]

    offset = 0
    if shaft=='A':
        electrode_amplifier_index_on_grid = np.array([3, 4, 2, 5, 1, 6, 0, 7])
        offset = 8
    elif shaft=='B':
        electrode_amplifier_index_on_grid = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        offset = 24
    elif shaft=='C':
        electrode_amplifier_index_on_grid = np.array([3, 7, 2, 6, 1, 5, 0, 4])
        offset = 16
    elif shaft=='D':
        electrode_amplifier_index_on_grid = np.array([3, 4, 2, 5, 1, 6, 0, 7])
        offset = 0

    electrode_amplifier_index_on_grid = electrode_amplifier_index_on_grid + offset

    electrode_amplifier_name_on_grid = np.array([shaft+str(p+1) for p, x in enumerate(electrode_amplifier_index_on_grid)])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions = channel_positions.reset_index()
    return channel_positions