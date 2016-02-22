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


def find_points_in_array_with_jitter(points_to_be_found, array_to_search, jitter_around_each_point):
    found_points = []
    indices_of_found_points_in_searched_array = []
    prev_spikes_added = 0
    curr_spikes_added = 0
    not_found_points = []
    index_of_klusta_spike_found = 0
    for juxta_spike in points_to_be_found[index_of_klusta_spike_found:]:
        for possible_extra_spike in np.arange(juxta_spike - jitter_around_each_point,
                                              juxta_spike + jitter_around_each_point):
            possible_positions = np.where(array_to_search == possible_extra_spike)[0]
            if len(possible_positions) != 0:
                index_of_klusta_spike_found = possible_positions[0]
                found_points.append(array_to_search[index_of_klusta_spike_found])
                indices_of_found_points_in_searched_array.append(index_of_klusta_spike_found)
                curr_spikes_added += 1
                break
        if curr_spikes_added > prev_spikes_added:
            prev_spikes_added = curr_spikes_added
        else:
            not_found_points.append(juxta_spike)
    print(np.shape(found_points))
    print(str(100 * (np.shape(found_points)[0] / len(points_to_be_found)))+'% found')
    return found_points, indices_of_found_points_in_searched_array, not_found_points
