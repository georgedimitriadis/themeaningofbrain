__author__ = 'George Dimitriadis'


import numpy as np
import pandas as pd
import itertools

def AKNP_V01_D_E100_Grid_Layout():
    electrode_coordinate_grid = list(itertools.product([1,2,3,4,5,6],[1,2,3,4,5,6]))
    electrode_amplifier_index_on_grid = [np.nan, 11, 14, 30, 27, np.nan,
                                              8, 5, 2, 18, 21, 24,
                                              7, 10, 13, 29, 26, 23,
                                              9, 4, 1, 17, 20, 25,
                                              6, 12, 15, 31, 28, 22,
                                        np.nan,  3, 0, 16, 19, np.nan]
    electrode_amplifier_name_on_grid = ["", "Int11", "Int14", "Int30", "Int27", "",
                                   "Int08", "Int05", "Int02", "Int18", "Int21", "Int24",
                                   "Int07", "Int10", "Int13", "Int29", "Int26", "Int23",
                                   "Int09", "Int04", "Int01", "Int17", "Int20", "Int25",
                                   "Int06", "Int12", "Int15", "Int31", "Int28", "Int22",
                                        "",  "Int03", "Int00", "Int16", "Int19", ""]
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    return channel_positions


def grid_layout_2times32channels():
    electrode_coordinate_grid = list(itertools.product(np.arange(1, 9), np.arange(1, 9)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    """
    electrode_amplifier_index_on_grid = np.array([11, 3, 19, 27, 43, 35, 51, 59,
                                                  4, 12, 28, 20, 36, 44, 60, 52,
                                                  10, 2, 18, 26, 42, 34, 50, 58,
                                                  5, 13, 29, 21, 37, 45, 61, 53,
                                                  9, 1, 17, 25, 41, 33, 49, 57,
                                                  6, 14, 30, 22, 38, 46, 62, 54,
                                                  8, 0, 16, 24, 40, 32, 48, 56,
                                                  7, 15, 31, 23, 39, 47, 63, 55])
    """
    electrode_amplifier_index_on_grid = np.array([20, 28, 12, 4, 52, 60, 44, 36,
                                                  27, 19, 3, 11, 59, 51, 35, 43,
                                                  21, 29, 13, 5, 53, 61, 45, 37,
                                                  26, 18, 2, 10, 58, 50, 34, 42,
                                                  22, 30, 14, 6, 54, 62, 46, 38,
                                                  25, 17, 1, 9, 57, 49, 33, 41,
                                                  23, 31, 15, 7, 55, 63, 47, 39,
                                                  24, 16, 0, 8, 56, 48, 32, 40])

    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    return channel_positions


def grid_layout_64channels_r_l(bad_channels=None):
    electrode_coordinate_grid = list(itertools.product(np.arange(1, 9), np.arange(1, 9)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = np.array([22, 25, 38, 41, 54, 57, 6, 9,
                                                  23, 24, 39, 40, 55, 56, 7, 8,
                                                  20, 27, 36, 43, 52, 59, 4, 11,
                                                  21, 26, 37, 42, 53, 58, 5, 10,
                                                  18, 29, 34, 45, 50, 61, 2, 13,
                                                  19, 28, 35, 44, 51, 60, 3, 12,
                                                  16, 31, 32, 47, 48, 63, 0, 15,
                                                  17, 30, 33, 46, 49, 62, 1, 14])
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions

def grid_layout_64channels_l_r(bad_channels=None):
    electrode_coordinate_grid = list(itertools.product(np.arange(1, 9), np.arange(1, 9)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = np.array([54, 57, 6, 9, 22, 25, 38, 41,
                                                  55, 56, 7, 8, 23, 24, 39, 40,
                                                  52, 59, 4, 11, 20, 27, 36, 43,
                                                  53, 58, 5, 10, 21, 26, 37, 42,
                                                  50, 61, 2, 13, 18, 29, 34, 45,
                                                  51, 60, 3, 12, 19, 28, 35, 44,
                                                  48, 63, 0, 15, 16, 31, 32, 47,
                                                  49, 62, 1, 14, 17, 30, 33, 46])
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions

def grid_layout_128channels_cr_cl_rr_rl(bad_channels=None):
    electrode_coordinate_grid = tuple(itertools.product(np.arange(1, 9), np.arange(1, 9))) + \
                                tuple(itertools.product(np.arange(9, 17), np.arange(1, 9)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid_front = np.array([22, 25, 38, 41, 54, 57, 6, 9,
                                                       23, 24, 39, 40, 55, 56, 7, 8,
                                                       20, 27, 36, 43, 52, 59, 4, 11,
                                                       21, 26, 37, 42, 53, 58, 5, 10,
                                                       18, 29, 34, 45, 50, 61, 2, 13,
                                                       19, 28, 35, 44, 51, 60, 3, 12,
                                                       16, 31, 32, 47, 48, 63, 0, 15,
                                                       17, 30, 33, 46, 49, 62, 1, 14])
    electrode_amplifier_index_on_grid_back = electrode_amplifier_index_on_grid_front + 64
    electrode_amplifier_index_on_grid = np.concatenate([electrode_amplifier_index_on_grid_front, electrode_amplifier_index_on_grid_back])
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions

def grid_layout_128channels_rl_rr_cl_cr(bad_channels=None):
    electrode_coordinate_grid = tuple(itertools.product(np.arange(1, 9), np.arange(1, 9))) + \
                                tuple(itertools.product(np.arange(9, 17), np.arange(1, 9)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid_back = np.array([54, 57, 6, 9, 22, 25, 38, 41,
                                                       55, 56, 7, 8, 23, 24, 39, 40,
                                                       52, 59, 4, 11, 20, 27, 36, 43,
                                                       53, 58, 5, 10, 21, 26, 37, 42,
                                                       50, 61, 2, 13, 18, 29, 34, 45,
                                                       51, 60, 3, 12, 19, 28, 35, 44,
                                                       48, 63, 0, 15, 16, 31, 32, 47,
                                                       49, 62, 1, 14, 17, 30, 33, 46])
    electrode_amplifier_index_on_grid_front = electrode_amplifier_index_on_grid_back + 64
    electrode_amplifier_index_on_grid = np.concatenate([electrode_amplifier_index_on_grid_front, electrode_amplifier_index_on_grid_back])
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions



def grid_layout_78channels_eric_inverse(bad_channels=None, top_14_channels_removed = False):
    electrode_coordinate_grid = list(itertools.product(np.arange(1, 11), np.arange(1, 11)))
    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = np.array([75, 93, 76, 92, 89,  0,  0,  0,  0,  0,
                                                  91, 74, 90, 73,  0,  0,  0,  0,  0,  0,
                                                  72, 88, 71,  0,  0, 25, 11, 13,  0,  0,
                                                  87, 70,  0,  0, 23,  9, 28, 30, 15, 48,
                                                  17,  2, 20,  5,  7, 26, 12, 14, 32, 64,
                                                   1, 19,  4, 22, 24, 10, 29, 31, 16, 47,
                                                  18,  3, 21,  6,  8, 27, 63, 46, 62, 45,
                                                   0, 49, 33, 51, 61, 44, 60, 43, 59, 42,
                                                   0,  0, 50, 35, 58, 41, 57, 40, 56, 39,
                                                   0,  0, 34, 52, 55, 38, 54, 37, 53, 36], dtype='int16')
    electrode_amplifier_name_on_grid = np.array(["CSC"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)
    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    channel_positions = channel_positions[channel_positions.Numbers!=0]
    if top_14_channels_removed:
        top_channels = [75, 93, 76, 92, 89, 91, 74, 90, 73, 72, 88, 71, 87, 70]
        channel_positions = channel_positions[~channel_positions.Numbers.isin(top_channels)]
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions

