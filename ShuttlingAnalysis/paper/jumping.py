# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:36:49 2014

@author: IntelligentSystem
"""

import os
import trajectories
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(path,ax=None):
    if ax is None:
        ax = plt.axes()
    traw = trajectories.trajectories(np.genfromtxt(os.path.join(path,'Analysis/trajectories.csv')))
    tcross = trajectories.crossings(traw)
    [plt.plot(tcross.data[s,0],600-tcross.data[s,1],color='k',alpha=0.1) for s in tcross.slices]
    
def plot_session_comparison(paths):
    for i,path in enumerate(paths):
        ax = plt.subplot(1,len(paths),i+1)
        plot_trajectories(path,ax)