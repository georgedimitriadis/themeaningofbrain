# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 11:42:00 2014

@author: Gonçalo
"""

import os
import cv2
import video
import siphon
import imgproc
import pltutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import activitytables
import activitymovies
from preprocess import labelpath, frames_per_second, max_width_cm
from preprocess import stepcenter_pixels, stepcenter_cm
from preprocess import slipcenter_pixels, slipcenter_cm
from collectionselector import CollectionSelector

def scatterhist(x,y,bins=10,color=None,histalpha=1,axes=None,
                xlim=None,ylim=None):
    if axes is None:
        axScatter = plt.subplot2grid((3,3),(1,0),rowspan=2,colspan=2)
        axHistx = plt.subplot2grid((3,3),(0,0),colspan=2)
        axHisty = plt.subplot2grid((3,3),(1,2),rowspan=2)
    else:
        axScatter,axHistx,axHisty = axes
    
    axScatter.scatter(x, y, c=color, edgecolors='none')
    if xlim is not None:
        axScatter.set_xlim(xlim)
    if ylim is not None:
        axScatter.set_ylim(ylim)
    
    # now determine nice limits by hand:
    axHistx.hist(x, bins=bins, range=xlim,alpha=histalpha,
                 color=color, edgecolor = 'none')
    axHisty.hist(y, bins=bins, range=ylim,orientation='horizontal',
                 color=color, alpha=histalpha, edgecolor = 'none')
    
    axHistx.set_xticks([])
    axHisty.set_yticks([])
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

def clusterroiframes(act,roiactivity,info,leftroi,rightroi,
                     roicenter_cm,cropframes):
    # Compute step times
    roidiff = roiactivity.diff()
    roipeaks = activitytables.findpeaks(roidiff,1500)
    pksloc = [[roidiff.index.get_loc(peak) for peak in roi] for roi in roipeaks]
    
    # Tile step frames    
    vidpaths = activitymovies.getmoviepath(info)
    timepaths = activitymovies.gettimepath(info)
    backpaths = activitymovies.getbackgroundpath(info)
    videos = [video.video(path,timepath) for path,timepath in zip(vidpaths,timepaths)]

    def getroiframes(roiindex,flip=False):
        roicenterxcm = roicenter_cm[roiindex][1]
        headdistance = [act.xhead[p] - roicenterxcm for p in pksloc[roiindex]]
        print headdistance
        framehead = [p for i,p in enumerate(pksloc[roiindex])
                     if (-25 < headdistance[i] < -5 if not flip
                     else 5 < headdistance[i] < 25)]
        
        frames = [cropframes(videos[0].frame(p),roiindex) for p in framehead]
        backgrounds = [cropframes(activitymovies.getbackground(backpaths[0],videos[0].timestamps[p]),roiindex)
                       for p in framehead]
        frames = [cv2.subtract(f,b) for f,b in zip(frames,backgrounds)]
        if flip:
            frames = [cv2.flip(f,1) for f in frames]
        return frames,framehead

    leftframes,leftindices = getroiframes(leftroi,False)
    rightframes,rightindices = getroiframes(rightroi,True)
    print "==========================="
    frames = np.array(leftframes + rightframes)
    frameindices = np.array(leftindices + rightindices)
    sortindices = np.argsort(frameindices)
    frames = frames[sortindices]
    frameindices = frameindices[sortindices]
    
    Z, R,labels,h = imgproc.cluster(frames,videos[0],frameindices)
    return frames,roidiff,roipeaks,pksloc,Z,R,labels

def clusterstepframes(act,info,leftstep,rightstep):
    stepactivity = act.iloc[:,17:25]
    return clusterroiframes(act,stepactivity,info,leftstep,rightstep,
                            stepcenter_cm,
                            lambda f,i:imgproc.croprect(stepcenter_pixels[i],
                                                        (200,200),f))
                            
def clusterslipframes(act,info,leftgap,rightgap):
    slipactivity = act.iloc[:,25:32]
    return clusterroiframes(act,slipactivity,info,leftgap,rightgap,
                            slipcenter_cm,
                            lambda f,i:imgproc.croprect((slipcenter_pixels[i][0]-100,slipcenter_pixels[i][1]),
                                                        (300,400),f))

def sessionmetric(data,connect=True,ax=None,colorcycle=None):
    if data.ndim != 2:
        raise ValueError("data must be two-dimensional table (value,error)")
        
    if data.index.nlevels != 3:
        raise ValueError("data has to be multi-level (session,lesion,subject)")
    
    if ax is None:
        ax = plt.gca()
    xticks = []
    groupcount = 0

    groupcenters = []
    groupmeans = []
    grouperr = []
    for session,subjectgroups in data.groupby(level=0):
        nsubjects = len(subjectgroups)
        step = 0.5 / nsubjects
        offset = -0.25
        ax.set_color_cycle(colorcycle)
        groupedsubjects = subjectgroups.groupby(level=1)
        groupcount = len(groupedsubjects)
        for i,(groupname,subjects) in enumerate(groupedsubjects):
            x = session + offset + step * np.arange(len(subjects))
            _,caps,_ = plt.errorbar(x,subjects.icol(0),subjects.icol(1),
                                      fmt=None,label=groupname,
                                      zorder=100,capthick=1,alpha=0.4)
            for cap in caps:
                cap.remove()

            if i >= len(groupcenters):
                groupcenters.append([])
                groupmeans.append([])
                grouperr.append([])
            groupcenters[i].append((x[-1] + x[0]) / 2.0)
            groupmeans[i].append(subjects.icol(0).mean())
            grouperr[i].append(subjects.icol(0).std())
            offset += step * len(subjects)
        xticks.append(session)
    
    ax.set_color_cycle(colorcycle)
    for center,mean,err in zip(groupcenters,groupmeans,grouperr):
        plt.errorbar(center,mean,err,
                     fmt='--' if connect else None,ecolor='k',
                     linewidth=2,capthick=2,markersize=0)
                     
    if not connect:
        ylims = ax.get_ylim()
        for left,right in zip(xticks,xticks[1:]):
            boundary = (left + right) / 2.0
            ax.plot((boundary,boundary),ylims,'k--')
        ax.set_ylim(ylims)
        
    ax.set_xticks(xticks)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:groupcount],labels[:groupcount])
    ax.set_xlabel(data.index.names[0])
        
def fpshist(activity,ax=None):
    if ax is None:
        ax = plt.gca()
    (1.0 / activity.timedelta).hist(ax=ax,bins=100,normed=True)
    ax.set_xlabel('fps')
    ax.set_title('frame rate')
    
def trajectoryplot(activity,crossings,ax=None,style='k',alpha=1,flip=False,
                   selector=lambda x:x.yhead,
                   ylabel = 'y (cm)'):
    if ax is None:
        ax = plt.gca()
    for s,side in crossings[['timeslice','side']].values:
        if flip and side == 'leftwards':
            ax.plot(max_width_cm - activity.xhead[s],
                    selector(activity)[s],style,alpha=alpha)
        else:
            ax.plot(activity.xhead[s],selector(activity)[s],style,alpha=alpha)
    ax.set_title('trajectories')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel(ylabel)
    
def featuresummary(crossings,ax=None,onselect=None):
    if ax is None:
        ax = plt.gca()
    pts = ax.scatter(crossings.duration,crossings.yhead_max,
               s=10,marker='D',facecolors='b',edgecolors='none')
    selector = CollectionSelector(ax,pts,color_other='r',
                                  onselection=onselect)
    ax.set_title('trajectory features')
    ax.set_xlabel('duration (s)')
    ax.set_ylabel('max height (cm)')
    #ax.set_ylabel('min speed (cm / s)')
    return selector
    
def slowdownsummary(crossings,ax=None,regress=True):
    if ax is None:
        ax = plt.gca()
    entryspeed = crossings.entryspeed
    exitspeed = crossings.exitspeed
    ax.plot(entryspeed,exitspeed,'.')
    if regress:
        pltutils.regressionline(entryspeed,exitspeed,ax,color='k')
    ax.set_title('slowdown')
    ax.set_xlabel('entry speed (cm / s)')
    ax.set_ylabel('exit speed (cm / s)')
    
def rewardrate(rewards,ax=None):
    if ax is None:
        ax = plt.gca()
    intervals = rewards.time.diff() / np.timedelta64(1,'m')
    ax.plot(1.0 / intervals)
    ax.set_title('reward rate')
    ax.set_xlabel('trials')
    ax.set_ylabel('r.p.m')
    
def clearhandles(handles):
    while len(handles) > 0:
        handle = handles.pop()
        if np.iterable(handle):
            for l in handle:
                l.remove()
                del l
        del handle
        
def clearcollection(handles):
    while len(handles) > 0:
        handle = handles.pop()
        handle.remove()
        del handle

def sessionsummary(path):
    labelh5path = labelpath(path)
    activity = activitytables.read_activity(path)
    crossings = activitytables.read_crossings(path,activity)
    rewards = activitytables.read_rewards(path)
    #steptimes = activitytables.steptimes(activity)
    vidpath = os.path.join(path,'front_video.avi')
    vid = video.video(vidpath)
    
    selected = []
    def onselect(ind):
        selector.ind[:] = ind
        selector.updateselection()
        clearhandles(selected)
        if len(ind) <= 0:
            return
        
        for s in crossings.slices[ind]:
            h = axs[0,1].plot(activity.xhead[s],activity.yhead[s],'r')
            selected.append(h)
            
    markers = []
    def updateplots():
        onselect([])
        clearcollection(markers)
        valid = crossings.label == 'valid'
        
        axs[0,1].clear()
        trajectoryplot(activity,crossings[valid],axs[0,1],alpha=0.2)
        
        axs[1,2].clear()
        slowdownsummary(crossings[valid],axs[1,2])
        
        axs[1,1].clear()
        
        
        invalid = crossings.label == 'invalid'
        if invalid.any():
            rows = crossings[invalid]
            pts = axs[0,2].scatter(rows.duration,rows.yhead_max,
                           s=20,marker='x',facecolors='none',edgecolors='r')
            markers.append(pts)
        fig.canvas.draw_idle()
            
    def onkeypress(evt):
        label = None
        if evt.key == 'q':
            crossings.label.to_hdf(labelh5path, 'label')
        if evt.key == 'x':
            label = 'invalid'
        if evt.key == 'c':
            label = 'valid'
        if evt.key == 'z' and len(selector.ind) == 1:
            frameslice = crossings.iloc[selector.ind[0],:].slices
            video.showmovie(vid,frameslice.start,
                            fps=frames_per_second,
                            frameend=frameslice.stop)
        if label != None:
            crossings.label[selector.ind] = label
            updateplots()
    
    fig, axs = plt.subplots(3,3)
    fpshist(activity,axs[0,0])
    selector = featuresummary(crossings,axs[0,2],onselect)
    updateplots()
    rewardrate(rewards,axs[1,0])
    fig.canvas.mpl_connect('key_press_event',onkeypress)
    
    plt.tight_layout()
    return activity,crossings,rewards,selector