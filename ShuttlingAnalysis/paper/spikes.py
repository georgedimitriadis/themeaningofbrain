# -*- coding: utf-8 -*-
"""
Created on Thu Sep 04 10:35:21 2014

@author: Gonçalo
"""

import numpy as np
import scipy.signal as signal

def findpeaks(x, threshold = 0, mingap = 0, axis = -1):
    # First derivative
    d = np.diff(x, axis = axis)
    
    # Sign changes of first derivative
    sd = np.diff(np.sign(d), axis = axis)
    
    # Zero-crossings of first derivative (up or down depending on threshold)
    zc = sd < 0 if threshold > 0 else sd > 0
    nz = np.nonzero(zc)
    
    # Shift indices to compensate for diff
    l = list(nz)
    l[axis] = l[axis] + 1
    nz = tuple(l)

    # Keep only threshold crossing peaks
    tp = x[nz] > threshold if threshold > 0 else x[nz] < threshold
    nz = np.array(nz)[:,tp]
    
    # Find groups of peaks separated by at least minimum gap
    pind = nz[axis,:]
    ndims = len(x.shape)
    dind = np.delete(nz, axis, axis = 0)
    if dind.shape[0] > 0:
        raise NotImplementedError("Dimensions higher than 1 are not supported.")
    
    ipi = np.diff(pind)
    section_indices = np.nonzero(ipi > mingap)[0] + 1
    sections = np.split(pind,section_indices)
    
    fsections = []
    for section in sections:
        adim = ndims - 1 if axis < 0 else axis
        indices = [np.s_[:] if i != adim else section for i in xrange(ndims)]
        values = x[indices]
        if threshold > 0:
            sind = np.argmax(values, axis = axis)
        else:
            sind = np.argmin(values, axis = axis)
        fsections.append(section[sind])

    return fsections

def findwaveforms(x,threshold,pre,post,axis=-1):
    if threshold >= 0:
        xthr = x > threshold
    else:
        xthr = x < threshold
        
    xthr = np.insert(np.int8(xthr), 0, 0, axis=axis)
    crossings = np.diff(xthr, axis) > 0
    