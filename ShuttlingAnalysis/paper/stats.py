# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:13:31 2014

@author: IntelligentSystem
"""

import numpy as np
import scipy.stats as stats

def meanstd(x,axis=None):
    return stats.nanmean(x,axis),stats.nanstd(x,axis)
    
def binning(fp,xp,bins=10):
    if not np.iterable(bins):
        if np.isscalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")
        range = (xp.min(), xp.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins+1, endpoint=True)
    else:
        bins = np.asarray(bins)
        if (np.diff(bins) < 0).any():
            raise AttributeError('bins must increase monotonically.')
    
    binindices = np.digitize(xp,bins)
    return np.bincount(binindices,fp) / np.bincount(binindices)