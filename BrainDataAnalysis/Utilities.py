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