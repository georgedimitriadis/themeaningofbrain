# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:44:37 2013

@author: gonca_000
"""

import dateutil
import numpy as np

class events:
    def __init__(self, timestamps):
        self.timestamps = timestamps
        
    def __add__(self, other):
        timestamps = np.concatenate((self.timestamps, other.timestamps))
        return events(np.sort(timestamps))
        
    def iei(self):
        return map(lambda x:x.total_seconds(),np.diff(self.timestamps))
        
def genfromtxt(path):
    timestamps = np.atleast_1d(np.genfromtxt(path, dtype=str))
    timestamps = [dateutil.parser.parse(xs) for xs in timestamps]
    return events(timestamps)