# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:25:29 2014

@author: IntelligentSystem
"""

import numpy as np

class steps:
    def __init__(self, data, threshold = 5000):
        self.data = data
        self.steps = np.insert(np.diff(np.int32(data > threshold),axis=0) > 0,0,False,0)
        
def preprocess(data,threshold,video):
    events = np.insert(np.diff(np.int32(data > threshold),axis=0) > 0,0,False,0)
    return [video.frame(evt) for evt in np.nonzero(events)[0]]