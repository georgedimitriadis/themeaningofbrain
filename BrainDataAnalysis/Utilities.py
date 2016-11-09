__author__ = 'IntelligentSystem'


import numpy as np


def find_closest(array, target):
    #a must be sorted
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array)-1)
    left = array[idx-1]
    right = array[idx]
    idx -= target - left < right - target
    return idx


def normList(L, normalizeFrom=0, normalizeTo=1):
    '''normalize values of a list to make its min = normalizeFrom and its max = normalizeTo'''
    vMax = max(L)
    vMin = min(L)
    return [(x-vMin)*(normalizeTo - normalizeFrom) / (vMax - vMin) for x in L]


def normList(L, normalizeFrom=0, normalizeTo=1, vMin=None, vMax=None):
    '''normalize values of a list to make its min = normalizeFrom and its max = normalizeTo'''
    if vMax:
        _vMax = vMax
    else:
        _vMax = max(L)

    if vMin:
        _vMin = vMin
    else:
        _vMin = min(L)

    return [(x-_vMin)*(normalizeTo - normalizeFrom) / (_vMax - _vMin) for x in L]


def nested_change(item, func):
    if isinstance(item, list):
        return [nested_change(x, func) for x in item]
    return func(item)


def find_points_in_array_with_jitter(array_of_points_to_be_found, array_to_search, jitter_around_each_point):
    found_points_in_aoptbf = []
    indices_of_found_points_in_aoptbf = []
    indices_of_found_points_in_searched_array = []
    prev_points_added = 0
    curr_points_added = 0
    not_found_points_in_aoptbf = []
    for index_of_aoptbf in np.arange(len(array_of_points_to_be_found)):
        point_to_be_found = array_of_points_to_be_found[index_of_aoptbf]
        for possible_point in np.arange(point_to_be_found - jitter_around_each_point,
                                        point_to_be_found + jitter_around_each_point):
            indices_of_possible_point_in_searched_array = np.where(array_to_search == possible_point)[0]
            if len(indices_of_possible_point_in_searched_array) != 0:
                found_points_in_aoptbf.append(array_to_search[indices_of_possible_point_in_searched_array[0]])
                indices_of_found_points_in_aoptbf.append(index_of_aoptbf)
                indices_of_found_points_in_searched_array.append(indices_of_possible_point_in_searched_array[0])
                curr_points_added += 1
                break
        if curr_points_added > prev_points_added:
            prev_points_added = curr_points_added
        else:
            not_found_points_in_aoptbf.append(point_to_be_found)
    print('Points found in array = ' + str(np.shape(found_points_in_aoptbf)[0]))
    print('Percentage = ' + str(100 * (np.shape(found_points_in_aoptbf)[0] / len(array_of_points_to_be_found))) + '% found')
    return found_points_in_aoptbf, indices_of_found_points_in_aoptbf, indices_of_found_points_in_searched_array, not_found_points_in_aoptbf
