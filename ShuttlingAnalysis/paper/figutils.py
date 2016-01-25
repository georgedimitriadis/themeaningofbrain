# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:15:12 2013

@author: gonca_000
"""

#import load_data
import numpy as np
import matplotlib.pyplot as plt
#import process_session as procses
import matplotlib.ticker as ticker
import pltutils
#import analysis_utilities as utils
import scipy.stats as stats
#import dateutil

numBins = 37
binSize =  27 # Approximately 25.6 pixels/cm (this needs to be adjusted for warping)

def plot_epoch_average(data,label=None,offset=0,scale=1):
    mean = [np.mean(epoch) for epoch in data]
    std = [np.std(epoch) for i,epoch in enumerate(data)]
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.IndexLocator(1,0))
    #c = next(ax._get_lines.color_cycle)
    x = (np.arange(len(data))*scale)+offset
    #plt.plot(x,mean,'--',zorder=0,color=c)
    (_,caps,_) = plt.errorbar(x,mean,std,fmt=None,label=label,zorder=100,capthick=1,alpha=0.4)
    for cap in caps:
        cap.remove()
    plt.xlim(-0.5-offset/2,(len(data)*scale)-0.5)
    
def get_bracket_y(measures,srange,c):
    return np.max([np.mean(measures[k][c])+np.std(measures[k][c]) for k in srange])
    
def get_negative_bracket_y(measures,srange,c):
    return np.min([np.mean(measures[k][c])-np.std(measures[k][c]) for k in srange])

#def plot_average_trial_times(name,sessiontimes,label=None,offset=0,scale=1,makefig=True):
#    trial_times = [[i.total_seconds() for i in session.inter_reward_intervals]
#                   if len(session.inter_reward_intervals) > 0
#                   else [procses.get_session_duration(session).total_seconds()]
#                   for session in sessions]
#    plot_epoch_average(trial_times,label,offset,scale)
#    return trial_times
    
def get_average_trial_time(session):
    return np.mean([d.total_seconds() for d in session.inter_reward_intervals])

def plot_comparison_measure(group_order,group_cycle,mclesionsham,bracket_offset=5,bracket_tickheight=1):
    scale = 46
    offsetscale = 2
    plt.close('all')
    fig = plt.figure('stable sessions average trial times',figsize=(4,7))
    ax = fig.gca()
    ax.set_color_cycle(group_cycle)
    measures = [plot_average_trial_times('stable sessions',mclesionsham[group_order[i]][1:5],label='lesion' if group_cycle[i] == 'r' else 'control',offset=i*2,scale=46) for i in range(14)]
    plt.ylabel('time between rewards (s)')
    plt.xlabel('sessions')
    
    lesionoffset = 3*offsetscale
    lesioncenter = [scale*i+lesionoffset for i in range(4)]
    lesion_times = [[get_average_trial_time(mclesionsham[i][s]) for i in group_order[0:7]] for s in range(1,5)]
    lesion_mean = [np.mean(x) for x in lesion_times]
    lesion_std = [np.std(x) for x in lesion_times]
    plt.errorbar(lesioncenter,lesion_mean,lesion_std,fmt='--o',color=group_cycle[0],ecolor='k',linewidth=2,capthick=2,markersize=0)
    
    controloffset = 10*offsetscale
    controlcenter = [scale*i+controloffset for i in range(4)]
    control_times = [[get_average_trial_time(mclesionsham[i][s]) for i in group_order[7:14]] for s in range(1,5)]
    control_mean = [np.mean(x) for x in control_times]
    control_std = [np.std(x) for x in control_times]
    plt.errorbar(controlcenter,control_mean,control_std,fmt='--o',color=group_cycle[7],ecolor='k',linewidth=2,capthick=2,markersize=0)
    
    pltutils.fix_font_size()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend((handles[0],handles[7]),(labels[0],labels[7]))
    
    tickoffset = 6.5*offsetscale
    xticks = [scale*i+tickoffset for i in range(4)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([1,2,3,4])
    
    ### BRACKETS (BETWEEN GROUPS) ###
    maxstd = []
    minstd = []
    significance = 0.05
    maxrange = len(measures)
    for i in range(len(xticks)):
        sigtest = stats.ttest_ind(lesion_times[i],control_times[i])[1]
        print sigtest,"groups"
        testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
        maxstd.append(get_bracket_y(measures,range(maxrange) if i < 1 else range(0,7)+range(8,14),i))
        minstd.append(get_negative_bracket_y(measures,range(maxrange) if i < 1 else range(0,7)+range(8,14),i))
        
    minstd = min(minstd)
    for i in range(len(xticks)):
        pltutils.hbracket(xticks[i],minstd-bracket_offset,1.4,label=testlabel,tickheight=-10*bracket_tickheight)
    #################################
    
    plt.draw()

def plot_trial_measures(group_cycle,group_order,trial_measures,bracket_offset=0.5,bracket_tickheight=1,ylims=None,xticklabels=None):
    scale = 46
    offsetscale = 2
    num_sessions = len(trial_measures[0])
    ax = plt.gca()
    ax.set_color_cycle(group_cycle)
    measures = [[trial_measures[group_order[i]][s] for s in (range(num_sessions) if i != 7 else [0])] for i in range(len(trial_measures))]
    [plot_epoch_average(measures[i],label='lesion' if group_cycle[i] == 'r' else 'control',offset=i*offsetscale,scale=scale) for i in range(len(trial_measures))]
    plt.ylabel('time to cross obstacles (s)')
    plt.xlabel('x')
    
    offset1 = 3*offsetscale
    center1 = [scale*i+offset1 for i in range(num_sessions)]
    group1_times = [[np.mean(trial_measures[i][s]) for i in group_order[0:7]] for s in range(num_sessions)]
    group1_mean = [np.mean(x) for x in group1_times]
    group1_std = [np.std(x) for x in group1_times]
    plt.errorbar(center1,group1_mean,group1_std,fmt='o',color=group_cycle[0],ecolor='k',linewidth=2,capthick=2,markersize=0)
    
    offset2 = 10*offsetscale
    center2 = [scale*i+offset2 for i in range(num_sessions)]
    group2_times = [[np.mean(trial_measures[i][s]) for i in (group_order[7:14] if s < 1 else group_order[8:14])] for s in range(num_sessions)]
    group2_mean = [np.mean(x) for x in group2_times]
    group2_std = [np.std(x) for x in group2_times]
    plt.errorbar(center2,group2_mean,group2_std,fmt='o',color=group_cycle[7],ecolor='k',linewidth=2,capthick=2,markersize=0)
    
    pltutils.fix_font_size()
    #handles, labels = ax.get_legend_handles_labels()
    #plt.legend((handles[0],handles[7]),(labels[0],labels[7]))
    tickoffset = 6.5*offsetscale
    xticks = [scale*i+tickoffset for i in range(num_sessions)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    
    ### BRACKETS (BETWEEN GROUPS) ###
    maxstd = []
    minstd = []
    significance = 0.01
    maxrange = len(measures)
    for i in range(len(xticks)):
        sigtest = stats.ttest_ind(group1_times[i],group2_times[i])[1]
        print sigtest,"groups"
        testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
        maxstd.append(get_bracket_y(measures,range(maxrange) if i < 1 else range(0,7)+range(8,14),i))
        minstd.append(get_negative_bracket_y(measures,range(maxrange) if i < 1 else range(0,7)+range(8,14),i))
        
    maxstd = max(maxstd)
    for i in range(len(xticks)):
        pltutils.hbracket(xticks[i],maxstd+bracket_offset,2,label=testlabel,tickheight=bracket_tickheight)
    ##################################
        
    ### BRACKETS (BETWEEN CONDITIONS) ###
    minstd = min(minstd)
    sigtest = stats.ttest_ind(group1_times[0]+group2_times[0],group1_times[-1]+group2_times[-1])[1]
    print sigtest,"conditions"
    testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
    #pltutils.hbracket(xticks[1],minstd-bracket_offset,5.5,label=testlabel,tickheight=-1.5*bracket_tickheight)
    pltutils.hbracket((xticks[-1]-xticks[0])/2+xticks[0],minstd-bracket_offset,9.5,label=testlabel,tickheight=-1.5*bracket_tickheight)
    #####################################
    
    ### SEPARATORS ###
    ylims = plt.ylim() if ylims is None else ylims
    for i in range(len(xticks)-1):
        separatorxi = (xticks[i]+xticks[i+1])/2
        ax.plot((separatorxi,separatorxi),ylims,'k--')
    plt.ylim(ylims)
    ##################
    
    plt.xlabel('')
    plt.draw()

############# FIGURE 3 ##############################

#### Get contact step activation ###
#def get_contact_step_activity(session,i):
#    activity = session.steps[i]
#    step_index = 3 if session.labels[i]['direction'] == 'right' else 4
#    return activity[:,step_index]
#    
#### Plot contact maximum activation distribution ###
#def plot_contact_distribution(session):
#    max_contacts = [np.max(get_contact_step_activity(session,i)) for i in range(len(session.steps))]
#    maxStContacts = [mc for state,mc in zip(session.labels,max_contacts) if state['state'] == 'stable']
#    maxUnContacts = [mc for state,mc in zip(session.labels,max_contacts) if state['state'] == 'unstable']
#    plt.plot(maxStContacts,'b.')
#    plt.plot(maxUnContacts,'r.')
#    
## Time Difference in microseconds
#def time_diff(t):
#    numTimes = np.size(t)    
#    dT = np.zeros(numTimes-1)    
#    for i in range(numTimes-1):
#        dT[i] = (t[i+1]-t[i]).microseconds
#
#    return dT
#    
#def get_randomized_speed_profiles(experiment,a,sessions,filterpath):    
#    avgSpeeds_allsess = []
#    trialTypes_allsess = []
#
#    # Select Session
#    for s in sessions:
#
#        # Load Valid Trial Filter (manually sorted)
#        validFilename = filterpath +  r'\valid_a' + str(a) + '_s' + str(s) + '.pickle'
#        valid_trials = load_data.load_pickle(validFilename)
#        numTrials = np.size(valid_trials)
#    
#        # There is some misalignment for sessions on the last 4 animals..there session 0 is other session 1
#        if a >= 10:
#            s = s-1
#        
#        # Set Trial Lables
#        labelfilter1 = {'state':'stable'}
#        labelfilter2 = {'state':'unstable'}
#        labelFilters = [labelfilter1, labelfilter2]
#        
#        # Look at all Trajectories
#        trajectories = experiment[a][s].trajectories
#        times = experiment[a][s].time    
#        slices = experiment[a][s].slices
#        labels = experiment[a][s].labels
#        steps = experiment[a][s].steps
#        speeds = experiment[a][s].speeds
#        
#        print str.format('a:s {0}:{1} {2} {3}', a, s, numTrials, len(slices))
#        
#        # Set Valid Trials (No exploration or tracking errors)
#        crossings = valid_trials
#    
#        # Set Binning and Range
#        avgSpeeds = np.zeros((numTrials, numBins))
#        trialTypes = np.zeros((numTrials, 1))
#        for t in range(0,numTrials):
#        
#            #label_indices = np.array(pt.get_labeled_indices(labels,labelFilters[l]))
#            c = crossings[t]
#            
#            # Load X Trajectories and flip all of 'Left'
#            trialX = trajectories[slices[c],0]
#            if utils.is_dict_subset({'direction':'left'},labels[c]):
#                # ALign on 2 important rails (the center of rail 3 is 550)
#                # and the centr of rail 4 is 737, therefore, the first encounter
#                # is at 550 going "right", and when flipped, (1280-737 = 543)
#                # going "left"...therefore, to correct for the shift, I subteact 1273 
#                # and align the left and right trials
#                trialX = np.abs(trialX-1273)
#                
#            # Load Y Trajectories
#            trialY = trajectories[slices[c],1]
#            
#            # Load and Parse Times
#            trialTstrings = times[slices[c]]
#            trialT = np.array([dateutil.parser.parse(timeString) for timeString in trialTstrings])
#            
#            # Measure Progression Speed
#            diffX =  np.diff(trialX)
#            diffT = time_diff(trialT)/1000000 # Time interval in seconds
#            speedX = np.concatenate((np.zeros(1) , diffX/diffT))
#        
#            # Find enter/exit and crop trials
#            indR = np.where(trialX > 1200)
#            indL = np.where(trialX < 150)
#            if (np.size(indR) > 0) and (np.size(indL) > 0):
#                exitInd = indR[0][0]+1
#                enterInd = indL[0][-1]
#                
#            trialX = trialX[enterInd:exitInd]
#            trialY = trialY[enterInd:exitInd]
#            speedX = speedX[enterInd:exitInd]
#            
#            # Bin (progrssion - X) Speed Profiles (from position 200 to 1200)
#            for b in range(0,numBins):
#                bins = np.where((trialX >= (200+(b*binSize))) & (trialX < (200+(b*binSize)+binSize)))
#                if np.size(bins) > 0:
#                    avgSpeeds[t, b] = np.mean(speedX[bins])
#                else:
#                    avgSpeeds[t, b] = np.NaN
#            
#            # Correct for starting speed - - first Third of assay
#            baseSpeed = stats.nanmean(avgSpeeds[t, 0:14])
#            avgSpeeds[t,:] = avgSpeeds[t,:]/baseSpeed
#            
#            # Get Lables
#            label = labels[c]
#            
#            if utils.is_dict_subset({'state':'stable'},label):
#                trialTypes[t] = 0
#            else:
#                trialTypes[t] = 1
#        
#        # Pool All Average Speeds/TrialTypes Across Sessions        
#        avgSpeeds_allsess.append(avgSpeeds)
#        trialTypes_allsess.append(trialTypes)
#    
#    avgSpeeds = np.concatenate(avgSpeeds_allsess)
#    trialTypes = np.concatenate(trialTypes_allsess)
#    return avgSpeeds,trialTypes
#
#def plot_randomized_speed_profiles(avgSpeeds,trialTypes):
#    # Set Plotting Attributes
#    color1 = (0.0, 0.0, 0.0, 0.1)
#    color2 = (1.0, 0.6, 0.0, 0.1)
#    color1b = (0.0, 0.0, 0.0, 1.0)
#    color2b = (1.0, 0.6, 0.0, 1.0)
#    
#    traceColors = [color1, color2]
#    boldColors = [color1b, color2b]    
#    
#    # Plot Average Speeds in bins
#    plt.figure()
#    numTrials = np.size(trialTypes)
#    for t in range(0,numTrials):
#        if trialTypes[t] == 0:
#            plt.plot(avgSpeeds[t,:], color=color1)
#        else:
#            plt.plot(avgSpeeds[t,:], color=color2)
#
#    stableTrials = np.where(trialTypes == 0)
#    unstableTrials = np.where(trialTypes == 1)
#    mSt = stats.nanmean(avgSpeeds[stableTrials, :], 1)
#    mUn = stats.nanmean(avgSpeeds[unstableTrials, :], 1)
#    eSt = stats.nanstd(avgSpeeds[stableTrials, :], 1)/np.sqrt(np.size(stableTrials)-1)
#    eUn = stats.nanstd(avgSpeeds[unstableTrials, :], 1)/np.sqrt(np.size(unstableTrials)-1)
#    
##    eSt = stats.nanstd(avgSpeeds[stableTrials, :], 1)
##    eUn = stats.nanstd(avgSpeeds[unstableTrials, :], 1)
#
#
#    mSt = mSt[0];    
#    mUn = mUn[0];    
#    eSt = eSt[0];    
#    eUn = eUn[0];
#    
#    plt.plot(mUn, color=color2b, linewidth = 7)
#    plt.plot(mSt, color=color1b, linewidth = 7)
#
##    plt.plot(mSt + eSt, color=color1b, linewidth = 0.5)
##    plt.plot(mSt - eSt, color=color1b, linewidth = 0.5)
##    plt.plot(mUn + eUn, color=color2b, linewidth = 0.5)
##    plt.plot(mUn - eUn, color=color2b, linewidth = 0.5)
#    #pltutils.fix_font_size()
#    plt.xlabel('crossing extent (cm)')
#    plt.ylabel('normalized horizontal speed')
#    pltutils.fix_font_size()
#    plt.axis([0, 39, 0, 3])
#    
#    
##### Figure 3b ####
#    
#def get_randomized_group_average_speed_profiles(profiles):        
#    stAvg = []
#    unAvg = []
#    stErr = []
#    unErr = []
#    
#    for avgSpeeds,trialTypes in profiles:
#        # Plot Average Speeds in bins
#        stableTrials = np.where(trialTypes == 0)
#        unstableTrials = np.where(trialTypes == 1)
#        
#        mSt = stats.nanmean(avgSpeeds[stableTrials, :], 1)
#        mUn = stats.nanmean(avgSpeeds[unstableTrials, :], 1)
#        eSt = stats.nanstd(avgSpeeds[stableTrials, :], 1)/np.sqrt(np.size(stableTrials)-1)
#        eUn = stats.nanstd(avgSpeeds[unstableTrials, :], 1)/np.sqrt(np.size(unstableTrials)-1)
#    
#        mSt = mSt[0]
#        mUn = mUn[0]
#        eSt = eSt[0]
#        eUn = eUn[0]
#    
#        stAvg.append(mSt)
#        unAvg.append(mUn)
#        stErr.append(eSt)
#        unErr.append(eUn)
#    return (stAvg,stErr),(unAvg,unErr)
#    
#def get_randomized_group_speed_profile_difference(avgProfiles):    
#    diffs = []
#    errors = []
#
#    # Unpack average profile structure
#    (stAvg,stErr),(unAvg,unErr) = avgProfiles
#    
#    for mSt,eSt,mUn,eUn in zip(stAvg,stErr,unAvg,unErr):
#        # Compute Difference Speed between Stable and Unstable Trials
#        mDiff = mUn-mSt
#        eDiff = np.sqrt((eSt*eSt) + (eUn*eUn))
#        diffs.append(mDiff)
#        errors.append(eDiff)
#        
#    diffs = np.array(diffs)
#    errors = np.array(errors)
#    return diffs,errors,np.mean(diffs[:, 20:], 1)
#    
#def plot_randomized_group_average_speed_profiles(avgProfiles,labelx=True,labely=True,legend=True,title=None):
#    # Set Plotting Attributes
#    color1b = (0.0, 0.0, 0.0, 1.0)
#    color2b = (1.0, 0.6, 0.0, 1.0)
#    
#    # Unpack average profile structure
#    (stAvg,stErr),(unAvg,unErr) = avgProfiles
#    
#    # Prepare Bulk Arrays
#    stAvg = np.array(stAvg)
#    unAvg = np.array(unAvg)
#    stErr = np.array(stErr)
#    unErr = np.array(unErr)
#    
#    # Plot Averge Speed Profiles
#    a1 = plt.plot(np.mean(unAvg,0), color = color2b, linewidth = 3,label='unstable')
#    a2 = plt.plot(np.mean(stAvg,0), color = color1b, linewidth = 3,label='stable')
#    if labelx:
#        plt.xlabel('crossing extent (cm)')
#    if labely:
#        plt.ylabel('normalized horizontal speed')
#    pltutils.fix_font_size()
#    plt.axis([0, numBins, 0.5, 2.0])
#    plt.yticks([0.75,1.0,1.25,1.5,1.75])
#    if legend:
#        plt.legend((a2[0],a1[0]),('stable','unstable'),loc='upper left')
#        
#    if title is not None:
#        ax = plt.gca()
#        ax.text(0.5,0.9,title,horizontalalignment='center',transform=ax.transAxes)
#    
#def plot_randomized_speed_profile_difference_comparison(controls,lesions):
#    controlMeans = np.mean(controls, 0)
#    controlMeanAll = np.mean(controlMeans)
#    controlError = np.std(controlMeans)/np.sqrt(7-1)
#    
#    lesionMeans = np.mean(lesions, 0)
#    lesionMeanAll = np.mean(lesionMeans)
#    lesionlError = np.std(lesionMeans)/np.sqrt(6-1)
#    
#    significance = 0.05
#    sigtest = stats.ttest_ind(controlMeans, lesionMeans)[1]
#    testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
#    print sigtest
#    
#    cX = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
#    lX = [1.9, 1.9, 1.9, 1.9, 1.9, 1.9]
#    
#    plt.plot([0,3], [0,0], color=[0.25,0.25,0.25,1], linewidth = 1)
#    plt.plot(cX, controlMeans, 'bo')
#    plt.plot(lX, lesionMeans, 'ro')
#    plt.plot(1.9, lesionMeans[5], 'o', color = [1.0, 0.75, 0.75, 1.0])
#    
#    #plt.bar(1.2, controlMeanAll, 0.2, color=[1.0,1.0,1.0,0.0])
#    plt.errorbar(1.3, controlMeanAll, controlError, marker='s', mfc='blue', ecolor = 'black', mec='black', ms=1, mew=1, capsize=5, elinewidth=2)
#    plt.plot([1.0,1.4], [controlMeanAll,controlMeanAll], color=[0.25,0.25,1.0,1], linewidth = 2)
#    
#    #plt.bar(1.6, lesionMeanAll, 0.2, color=[1.0,1.0,1.0,0.0])
#    plt.errorbar(1.7, lesionMeanAll, lesionlError, marker='s', mfc='red', ecolor = 'black', mec='black', ms=1, mew=1, capsize=5, elinewidth=2)
#    plt.plot([1.6,2.0], [lesionMeanAll,lesionMeanAll], color=[1.00,0.25,0.25,1], linewidth = 2)
#    pltutils.hbracket(1.5,0.17,4.5,label=testlabel,tickheight=0.01)
#    
#    ax = plt.gca()
#    ax.set_xticks([1.1,1.9])
#    ax.set_xticklabels(['controls','lesions'])
#    plt.ylabel('normalized speed difference')
#    
#    pltutils.fix_font_size()
#    plt.axis([0.75,2.25,-0.2,0.2])