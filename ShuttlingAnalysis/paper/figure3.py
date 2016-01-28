# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:01:25 2014

@author: Gonçalo
"""

import os
import trials
import preprocess
import trajectories
import numpy as np
import scipy.stats as stats
import dateutil.parser as tparser
import matplotlib.pyplot as plt

numBins = 37
binSize =  27 # Approximately 25.6 pixels/cm (this needs to be adjusted for warping)

def get_randomized_speed_profiles(sessions):    
    avgSpeeds_allsess = []
    trialTypes_allsess = []

    # Select Session
    for s in sessions:
        
        # Load trials
        rawtrials = trajectories.genfromtxt(s)
        validtrials = trajectories.heightfilter(rawtrials,430,590)
        [plt.plot(t[:,0],t[:,1],'k',alpha=0.1) for t in validtrials.tolist()]
        plt.gca().invert_yaxis()
        numTrials = validtrials.slices.shape[0]
        print numTrials
    
        # There is some misalignment for sessions on the last 4 animals..there session 0 is other session 1
#        if a >= 10:
#            s = s-1
        
        # Look at all Trajectories
        traj = validtrials
        preprocess.make_videoanalysis(os.path.join(s,'Analysis'))
        vtimepath = os.path.join(s, 'Analysis/videotime.csv')
        vtime = np.genfromtxt(vtimepath)
        trajtrials = trials.gettrialindices(s)
        stepstate = trials.gettrialstate(os.path.join(s,'step3_trials.csv'),trajtrials)
        
        #print str.format('a:s {0}:{1} {2} {3}', a, s, numTrials, len(validtrials.slices))
        
        # Set Valid Trials (No exploration or tracking errors)
        crossings = traj
    
        # Set Binning and Range
        avgSpeeds = np.zeros((numTrials, numBins))
        trialTypes = np.zeros((numTrials, 1))
        for t in range(0,numTrials):
        
            #label_indices = np.array(pt.get_labeled_indices(labels,labelFilters[l]))
            #c = crossings[t]
            
            # Load X Trajectories and flip all of 'Left'
            trialX = crossings.data[crossings.slices[t],0]
            if trialX[0] > trialX[-1]:
                # ALign on 2 important rails (the center of rail 3 is 550)
                # and the centr of rail 4 is 737, therefore, the first encounter
                # is at 550 going "right", and when flipped, (1280-737 = 543)
                # going "left"...therefore, to correct for the shift, I subteact 1273 
                # and align the left and right trials
                trialX = np.abs(trialX-1273)
                
            # Load Y Trajectories
            trialY = crossings.data[crossings.slices[t],1]
            
            # Load and Parse Times
            trialT = vtime[crossings.slices[t]]
            
            # Measure Progression Speed
            diffX = np.diff(trialX)
            diffT = np.diff(trialT) # Time interval in seconds
            speedX = np.concatenate((np.zeros(1) , diffX/diffT))
        
            # Find enter/exit and crop trials
#            indR = np.where(trialX > 1200)
#            indL = np.where(trialX < 150)
#            if (np.size(indR) > 0) and (np.size(indL) > 0):
#                exitInd = indR[0][0]+1
#                enterInd = indL[0][-1]
#                
#            trialX = trialX[enterInd:exitInd]
#            trialY = trialY[enterInd:exitInd]
#            speedX = speedX[enterInd:exitInd]
            
            # Bin (progrssion - X) Speed Profiles (from position 200 to 1200)
            for b in range(0,numBins):
                bins = np.where((trialX >= (200+(b*binSize))) & (trialX < (200+(b*binSize)+binSize)))
                if np.size(bins) > 0:
                    avgSpeeds[t, b] = np.mean(speedX[bins])
                else:
                    avgSpeeds[t, b] = np.NaN
            
            # Correct for starting speed - - first Third of assay
            baseSpeed = stats.nanmean(avgSpeeds[t, 0:14])
            avgSpeeds[t,:] = avgSpeeds[t,:]/baseSpeed
            
            # Get Lables            
            if stepstate[crossings.slices[t].start]:
                trialTypes[t] = 0
            else:
                trialTypes[t] = 1
        
        # Pool All Average Speeds/TrialTypes Across Sessions        
        avgSpeeds_allsess.append(avgSpeeds)
        trialTypes_allsess.append(trialTypes)
    
    avgSpeeds = np.concatenate(avgSpeeds_allsess)
    trialTypes = np.concatenate(trialTypes_allsess)
    return avgSpeeds,trialTypes

def plot_randomized_speed_profiles(avgSpeeds,trialTypes):
    # Set Plotting Attributes
    color1 = (0.0, 0.0, 0.0, 0.1)
    color2 = (1.0, 0.6, 0.0, 0.1)
    color1b = (0.0, 0.0, 0.0, 1.0)
    color2b = (1.0, 0.6, 0.0, 1.0)
    
    traceColors = [color1, color2]
    boldColors = [color1b, color2b]    
    
    # Plot Average Speeds in bins
    plt.figure()
    numTrials = np.size(trialTypes)
    for t in range(0,numTrials):
        if trialTypes[t] == 0:
            plt.plot(avgSpeeds[t,:], color=color1)
        else:
            plt.plot(avgSpeeds[t,:], color=color2)

    stableTrials = np.where(trialTypes == 0)
    unstableTrials = np.where(trialTypes == 1)
    mSt = stats.nanmean(avgSpeeds[stableTrials, :], 1)
    mUn = stats.nanmean(avgSpeeds[unstableTrials, :], 1)
    eSt = stats.nanstd(avgSpeeds[stableTrials, :], 1)/np.sqrt(np.size(stableTrials)-1)
    eUn = stats.nanstd(avgSpeeds[unstableTrials, :], 1)/np.sqrt(np.size(unstableTrials)-1)
    
#    eSt = stats.nanstd(avgSpeeds[stableTrials, :], 1)
#    eUn = stats.nanstd(avgSpeeds[unstableTrials, :], 1)


    mSt = mSt[0];    
    mUn = mUn[0];    
    eSt = eSt[0];    
    eUn = eUn[0];
    
    plt.plot(mUn, color=color2b, linewidth = 7)
    plt.plot(mSt, color=color1b, linewidth = 7)

#    plt.plot(mSt + eSt, color=color1b, linewidth = 0.5)
#    plt.plot(mSt - eSt, color=color1b, linewidth = 0.5)
#    plt.plot(mUn + eUn, color=color2b, linewidth = 0.5)
#    plt.plot(mUn - eUn, color=color2b, linewidth = 0.5)
    #pltutils.fix_font_size()
    plt.xlabel('crossing extent (cm)')
    plt.ylabel('normalized horizontal speed')
    #pltutils.fix_font_size()
    #plt.axis([0, 39, 0, 3])
    
def timedelta(data):
    return np.array([delta.total_seconds()
    for delta in np.diff(np.array([tparser.parse(d) for d in data]))])

def figure3(paths):
    stable = []
    unstable = []
    for path in paths:
        
        traj = trajectories.genfromtxt(path)
        steps = np.genfromtxt(os.path.join(path,'Analysis\step_activity.csv'))
        time = np.genfromtxt(os.path.join(path,'front_video.csv'),dtype=str)
        trajtrials = trials.gettrialindices(path)
        stepstate = trials.gettrialstate(os.path.join(path,'step3_trials.csv'),trajtrials)
        
        ftraj = trajectories.heightfilter(trajectories.lengthfilter(traj,0,500),0,6)
        
    #    plt.figure()
    #    for t in traj.tolist():
    #        plt.plot(len(t),max(t[:,1]),'k.')
    #    for t in ftraj.tolist():
    #        plt.plot(len(t),max(t[:,1]),'r.')
        
        traj = trajectories.mirrorleft(ftraj)
        #activesteptrials = [sum(steps[s,3 if s.step < 0 else 2]) / 255 for s in traj.slices]
        activesteptrials = [s for s in traj.slices
        if (sum(steps[s,3 if s.step < 0 else 2]) / 255) > 500]
        traj = trajectories.trajectories(traj.data,activesteptrials)
    
    #    plt.figure()    
    #    for t in traj.tolist():
    #        plt.plot(t[:,0],t[:,1],'k',alpha=0.1)
        
        speed = [np.insert(np.diff(traj.data[s,0])/timedelta(time[s]),0,0)
        for s in traj.slices]
        validtrials = [traj.slices[i] for i,s in enumerate(speed) if np.mean(s) > 0]
        
        traj = trajectories.trajectories(traj.data,validtrials)
        speed = [s for s in speed if np.mean(s) > 0]
        
    #    plt.figure()
    #    for sp in speed:
    #        plt.plot(sp,'k',alpha=0.1)
            
        # Bin (progrssion - X) Speed Profiles (from position 200 to 1200)
        numBins = 25
        binSize = 50 / numBins
        bins = range(0,50,binSize)
        speedbins = np.zeros((len(traj.slices),numBins))
        for i,t in enumerate(traj.tolist()):
            xs = t[:,0]
            binindices = np.digitize(xs,bins)
            for b in range(numBins):
                speedbin = speed[i][binindices == b]
                speedbins[i,b] = np.mean(speedbin) if speedbin.size > 0 else np.nan
            basespeed = stats.nanmean(speedbins[i,0:numBins/3])
            speedbins[i,:] /= basespeed
        
        stabletrials = [i for i,s in enumerate(validtrials) if stepstate[s.start]]
        unstabletrials = [i for i,s in enumerate(validtrials) if not stepstate[s.start]]
        
        stablespeeds = speedbins[stabletrials,:]
        unstablespeeds = speedbins[unstabletrials,:]
        stable.append(stablespeeds)
        unstable.append(unstablespeeds)
        
    def plotcurve(speeds,color):
        x = np.arange(numBins) * binSize
        mu = np.mean(speeds,axis=0)
        sd = np.std(speeds,axis=0)
        plt.plot(x,mu,color,linewidth=3)
        plt.plot(x,mu-sd,'k--')
        plt.plot(x,mu+sd,'k--')
        plt.fill_between(x,mu-sd,mu+sd,alpha=0.1,color=color)
    
    plt.figure()
    stablespeeds = np.concatenate(stable,0)
    unstablespeeds = np.concatenate(unstable,0)
    plotcurve(stablespeeds,'g')
    plotcurve(unstablespeeds,'r')
    return stablespeeds,unstablespeeds