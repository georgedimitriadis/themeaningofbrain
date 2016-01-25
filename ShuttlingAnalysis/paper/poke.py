# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:43:43 2013

@author: gonca_000
"""

import dateutil
import numpy as np

class poke:
    def __init__(self, activity, timestamps, rewards):
        self.activity = activity
        self.timestamps = timestamps
        self.rewards = rewards
        
    def __add__(self, other):
        activity = np.concatenate((self.activity, other.activity))
        timestamps = np.concatenate((self.timestamps, other.timestamps))
        timeindices = np.argsort(timestamps)
        rewards = np.sort(np.concatenate((self.rewards, other.rewards)))
        return poke(activity[timeindices],timestamps[timeindices],rewards)

def genfromtxt(pokepath, rewardpath):
    activity = np.atleast_1d(np.genfromtxt(pokepath, usecols=0))
    timestamps = np.atleast_1d(np.genfromtxt(pokepath, usecols=1, dtype=str))
    rewards = [dateutil.parser.parse(xs) for xs in np.atleast_1d(np.genfromtxt(rewardpath, dtype=str))]
    return poke(activity, timestamps, rewards)