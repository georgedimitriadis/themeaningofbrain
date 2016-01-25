# -*- coding: utf-8 -*-
"""
Created on Tue Sep 02 12:30:20 2014

@author: Gonçalo
"""

import os
import trials
import preprocess
import trajectories
import numpy as np
import matplotlib.pyplot as plt

# Load trajectories
path = r'D:/Protocols/Behavior/Shuttling/LightDarkServoStable/Data/JPAK_20/2013_04_04-11_40'
path = r'D:/Protocols/Behavior/Shuttling/LightDarkServoStable/Data/JPAK_21/2013_04_04-12_14'
path = r'D:/Protocols/Behavior/Shuttling/LightDarkServoStable/Data/JPAK_21/2013_04_16-11_35'
path = r'D:/Protocols/Behavior/Shuttling/LightDarkServoStable/Data/JPAK_23/2013_04_23-14_49'
#path = r'D:/Random/Data/JPAK_21/2013_04_16-11_35'
preprocess.make_videoanalysis(os.path.join(path,'Analysis'))
traj = trajectories.genfromtxt(path)
traj = trajectories.scale(traj)

# Plot trajectories
#p = [plt.plot(t[:,0],t[:,1]) for t in traj.tolist()]

# Load video time
vtimepath = os.path.join(path, 'Analysis/videotime.csv')
vtime = np.genfromtxt(vtimepath)

# Filter trajectories by height
ftraj = trajectories.heightfilter(traj,0,5)
print ftraj.slices.shape

# Filter trajectories by step activity
#steps = np.genfromtxt(os.path.join(path,'Analysis\step_activity.csv'))

# Compute speed and mirror values for left trials
sp = trajectories.speed(ftraj,vtime)
mtraj = trajectories.mirrorleft(ftraj)

for s in mtraj.slices:
    if s.start < s.stop: # right
        x = mtraj.data[s,0]
        y = mtraj.data[s,1]
    else:
        x = mtraj.data[s,0]-7*trajectories.width_pixel_to_cm
        y = mtraj.data[s,1]
    plt.plot(x,y,'k',alpha=0.1)

bins = np.linspace(0,40,40)
rawspeedbins = np.array(trajectories.speedbins(mtraj,sp,bins))

# Compute baseline speed in 1st third of assay and subtract
baselinesp = np.nanmean(rawspeedbins[:,0:rawspeedbins.shape[1]/3],axis=1)
speedbins = rawspeedbins-baselinesp.reshape((len(baselinesp),1))

# Get step state
trajtrials = trials.gettrialindices(path)
stepstate = trials.gettrialstate(os.path.join(path,'step3_trials.csv'),trajtrials)

# Split trials into stable/unstable
validtrials = mtraj.slices
stabletrials = [i for i,s in enumerate(validtrials) if stepstate[s.start]]
unstabletrials = [i for i,s in enumerate(validtrials) if not stepstate[s.start]]

plt.figure()
p = [plt.plot(t,'g',alpha=0.1) for t in speedbins[stabletrials]]
p = [plt.plot(t,'r',alpha=0.1) for t in speedbins[unstabletrials]]
m = np.nanmean(speedbins[stabletrials],axis=0)
plt.plot(m,'g')
m = np.nanmean(speedbins[unstabletrials],axis=0)
plt.plot(m,'r')
plt.show()