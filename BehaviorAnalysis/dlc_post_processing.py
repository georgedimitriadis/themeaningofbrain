
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.formula.api import ols


def assign_croped_markers_to_full_arena(markers_croped, crop_window_position):
    num_of_columns_to_change = \
        markers_croped.loc[:, markers_croped.columns.get_level_values(2).isin(['x'])].values.shape[1]

    markers = markers_croped.copy()

    markers.loc[:, markers_croped.columns.get_level_values(2).isin(['x'])] = \
        markers_croped.loc[:, markers_croped.columns.get_level_values(2).isin(['x'])].values + \
        np.repeat(np.expand_dims(crop_window_position['X'].values, axis=1), num_of_columns_to_change, axis=1)

    markers.loc[:, markers_croped.columns.get_level_values(2).isin(['y'])] = \
        markers_croped.loc[:, markers_croped.columns.get_level_values(2).isin(['y'])].values + \
        np.repeat(np.expand_dims(crop_window_position['Y'].values, axis=1), num_of_columns_to_change, axis=1)

    return markers


def clean_large_movements(positions, maximum_pixels):
    for i in np.arange(positions.shape[0]) + 1:
        try:
            '''
            if np.abs(positions[i - 1, 0] - positions[i, 0]) > maximum_pixels or \
                    np.isnan(np.abs(positions[i - 1, 0] - positions[i, 0])) or \
                    np.abs(positions[i - 1, 1] - positions[i, 1]) > maximum_pixels or \
                    np.isnan(np.abs(positions[i - 1, 1] - positions[i, 1])):
                positions[i, :] = positions[i - 1, :]
            '''
            distance = np.sqrt(np.power(positions[i - 1, 0] - positions[i, 0], 2) -
                               np.power(positions[i - 1, 1] - positions[i, 1], 2))
            if distance > maximum_pixels:
                positions[i, :] = positions[i - 1, :]
        except:
            pass

    return positions


def clean_large_movements_single_axis(positions, maximum_pixels):
    new_positions = np.copy(positions)
    for i in np.arange(positions.shape[0]) + 1:
        try:
            if np.abs(positions[i, 0] - positions[i - 1, 0]) > maximum_pixels:
                new_positions[i, 0] = positions[i + 1, 0]
            if np.abs(positions[i, 1] - positions[i + 1, 1]) > maximum_pixels:
                new_positions[i, 1] = positions[i - 1, 1]
        except:
            pass

    return new_positions


def turn_low_likelihood_to_nans(markers, likelihood_threshold):
    body_parts = np.unique(markers.columns.get_level_values(1))

    for body_part in body_parts:
        indices_of_low_likelihood = markers.loc[:,
                                    np.logical_and(markers.columns.get_level_values(1).isin([body_part]),
                                                   markers.columns.get_level_values(2).isin(['likelihood']))] < \
                                    likelihood_threshold

        markers.loc[indices_of_low_likelihood.values[:, 0], np.logical_and(
            markers.columns.get_level_values(1).isin([body_part]),
            markers.columns.get_level_values(2).isin(['x', 'y']))] = np.nan

    return markers


def seperate_markers(markers, body_parts):
    body_markers = markers.loc[:, markers.columns.get_level_values(1).isin(body_parts)]

    return body_markers


def average_multiple_markers_to_single_one(markers, flip=True):
    positions = np.array(
        [np.nanmean(markers.loc[:, markers.columns.get_level_values(2) == 'x'].values, axis=1),
         np.nanmean(markers.loc[:, markers.columns.get_level_values(2) == 'y'].values, axis=1)]). \
        transpose()

    if flip:
        positions[:, 1] = np.abs(positions[:, 1] - 640)

    return positions


def find_windows_with_nans(positions, gap):
    i = 0
    previous_end = None
    windows = []
    while i < len(positions) - 1:
        if np.isnan(positions[i]):
            start = i - 1
            try:
                while np.isnan(positions[i]):
                    i += 1
            except KeyError:
                break
            end = i
            if previous_end is not None:
                if start - previous_end < gap:
                    start = previous_start
                    windows.pop(-1)
            windows.append((start, end))
            previous_start = start
            previous_end = end
        i += 1
    print('Num of windows found = {}'.format(str(len(windows))))
    return windows


def get_data_from_a_nan_window(positions, window, gap):
    start = window[0] - gap + 1
    end = window[1] + gap - 1
    data = positions.loc[start:end]

    return data


def linear_interpolate_data_with_nans(data):
    x = np.arange(len(data))
    x = sm.add_constant(x)
    model = sm.OLS(data, x, missing='drop')
    fit = model.fit()
    x = x[:, 1]
    interpolate = x * fit.params['x1'] + fit.params['const']
    rsquared = fit.rsquared
    return interpolate, rsquared


def poly_interpolate_data_with_nans(data, order):
    x = np.arange(len(data))
    assert order > 1, 'Use linear_interpolate_data_with_nans if order is not > 1'
    data = pd.DataFrame({'x': x, 'y': data})

    formula = 'I(x**2) + x'
    for i in range(3, order + 1, 1):
        formula = 'I(x**{}) + '.format(str(i)) + formula
    formula = 'y ~ ' + formula

    model = ols(formula=formula, data=data)
    fit = model.fit()
    interpolate = np.power(x, 2) * fit.params['I(x ** 2)'] + x * fit.params['x'] + fit.params['Intercept']
    for i in range(3, order + 1, 1):
        interpolate += np.power(x, i) * fit.params['I(x ** {})'.format(str(i))]

    rsquared = fit.rsquared
    return interpolate, rsquared


def transform_to_interpolate(window_index, windows, figure, positions, gap, order=1):
    """
    Used for generating on the fly guis

    :param window_index:
    :param windows:
    :param figure:
    :param positions:
    :param gap:
    :param order:
    :return:
    """
    figure.clear()
    ax = figure.add_subplot(111)

    data = get_data_from_a_nan_window(positions, windows[window_index], gap)
    x = np.arange(len(data))

    ax.plot(x, data)
    if order > 1:
        interpolate, rsquared = poly_interpolate_data_with_nans(data, order)
    else:
        interpolate, rsquared = linear_interpolate_data_with_nans(data)
    ax.plot(x, interpolate)

    middle = interpolate[int(len(data)/2)]

    return middle, rsquared


def clean_dlc_outpout(updated_markers_filename, markers, gap, order):
    updated_markers = markers.copy()
    for c in np.arange(len(updated_markers.columns)):
        # Get the column of positions and make sure the 1st element is not nan
        positions = updated_markers.loc[:, updated_markers.columns[c]]
        print('Doing column {}'.format(updated_markers.columns[c]))
        if np.isnan(positions[0]):
            positions.loc[0] = positions.loc[1]

        # Get the windows that have nan in them
        windows = find_windows_with_nans(positions, gap=gap)

        index = 0
        num_of_windows = len(windows)
        for window in windows:
            data = get_data_from_a_nan_window(positions, window, gap)
            x = np.arange(len(data))
            if order < 2:
                interpolate, rsquared = poly_interpolate_data_with_nans(data, order)
            else:
                interpolate, rsquared = linear_interpolate_data_with_nans(data)
            for i in np.arange(window[0], window[1], 1):
                if np.isnan(positions.loc[i]):
                    x = i - window[0]
                    positions.loc[i] = interpolate[x]
            print('    {} of {} for {} of {}'.format(str(index), str(num_of_windows), str(c),
                                                     str(len(markers.columns))))
            index += 1

        updated_markers.loc[:, updated_markers.columns[c]] = positions

    pd.to_pickle(updated_markers, updated_markers_filename)

    return updated_markers
