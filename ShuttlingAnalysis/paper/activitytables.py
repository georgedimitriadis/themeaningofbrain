# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 05:22:49 2014

@author: Gonçalo
"""

import os
import cv2
import video
import imgproc
import sessions
import numpy as np
import pandas as pd
import activitymovies
import scipy.stats as stats
from scipy.interpolate import interp1d
from preprocess import storepath, labelpath
from preprocess import frontactivity_key, rewards_key, info_key
from preprocess import max_width_cm, width_pixel_to_cm
from preprocess import rail_start_pixels, rail_stop_pixels
from preprocess import stepcenter_cm, slipcenter_cm
from preprocess import stepcenter_pixels, slipcenter_pixels
from preprocess import rail_start_cm, rail_stop_cm

def groupbylesionvolumes(data,info):
    lesionvolume = info['lesionleft'] + info['lesionright']
    lesionvolume.name = 'lesionvolume'
    g = pd.concat([data,lesionvolume,info['cagemate']],axis=1)
    #joininfo = pd.concat((lesionvolume,info['cagemate']),axis=1)
    #g = data.join(joininfo)
    lesionorder = g[g['lesionvolume'] > 0].sort('lesionvolume',ascending=False)
    controls = lesionorder.groupby('cagemate',sort=False).median().index
    controls.name = 'subject' # OPTIONAL?
    controlorder = g.reset_index().set_index('subject').ix[controls]
    controlorder.set_index('session',append=True,inplace=True)
    
    result = pd.concat([controlorder,lesionorder])
    result['lesion'] = ['lesion' if v > 0 else 'control'
                        for v in result['lesionvolume']]
    result.reset_index(inplace=True)
    result.sort(['session','lesion'],inplace=True)
    result.set_index(['session','lesion','subject'],inplace=True)
    result.drop(['lesionvolume','cagemate'],axis=1,inplace=True)
    return result

def geomediancost(median,xs):
    return np.linalg.norm(xs-median,axis=1).sum()
    
def mad(xs):
    median = xs.median()
    return (xs - median).abs().median()
    
def read_activity(path):
    return pd.read_hdf(storepath(path), frontactivity_key)
    
def read_rewards(path):
    return pd.read_hdf(storepath(path), rewards_key)
    
def read_crossings(path, activity):
    crosses = crossings(activity)
    labelh5path = labelpath(path)
    if os.path.exists(labelh5path):
        crosses.label = pd.read_hdf(labelh5path, 'label')
    return crosses
    
def read_crossings_group(folders):
    crossings = []
    for path in folders:
        activity = read_activity(path)
        cr = read_crossings(path, activity)
        cr['session'] = os.path.split(path)[1]
        crossings.append(cr)
    return pd.concat(crossings)
    
def appendlabels(data,labelspath):
    if os.path.exists(labelspath):
        with open(labelspath) as f:
            for line in f:
                label,value = line.split(':')
                try:
                    value = float(value)
                except ValueError:
                    value = value
                data[label] = value
    
def read_subjects(folders, days=None,
                  key=frontactivity_key, selector=None,includeinfokey=True):
    if isinstance(folders, str):
        folders = [folders]
                      
    subjects = []
    for path in folders:
        subject = read_sessions(sessions.findsessions(path, days),
                                key,selector,includeinfokey)
        subjects.append(subject)
    return pd.concat(subjects)
    
def read_sessions(folders, key=frontactivity_key, selector=None,
                  includeinfokey=True):
    if isinstance(folders, str):
        folders = [folders]
    
    sessions = []
    for path in folders:
        session = pd.read_hdf(storepath(path), key)
        if selector is not None:
            session = selector(session)

        if key != info_key and includeinfokey:
            info = pd.read_hdf(storepath(path), info_key)
            info.reset_index(inplace=True)
            keys = [n for n in session.index.names if n is not None]
            session.reset_index(inplace=True)
            session['subject'] = info.subject.iloc[0]
            session['session'] = info.session.iloc[0]
            session.set_index(['subject', 'session'], inplace=True)
            session.set_index(keys, append=True, inplace=True)
        sessions.append(session)
    return pd.concat(sessions)
    
def slowdown(crossings):
    return pd.DataFrame(
    [stats.linregress(crossings.entryspeed,crossings.exitspeed)],
     columns=['slope','intercept','r-value','p-value','stderr'])
     
def findpeaks(ts,thresh,axis=-1):
    valid = ts > thresh if thresh > 0 else ts < thresh
    masked = np.ma.masked_where(valid,ts)

    views = np.rollaxis(masked,axis) if ts.ndim > 1 else [masked]
    clumpedpeaks = []
    for i,view in enumerate(views):
        clumped = np.ma.clump_masked(view)
        peaks = [ts[slce].ix[:,i].argmax() if thresh > 0 else ts[slce].ix[:,i].argmin()
                 for slce in clumped]
        clumpedpeaks.append(peaks)
    return clumpedpeaks if len(clumpedpeaks) > 1 else clumpedpeaks[0]
     
def roiactivations(roiactivity,thresh,roicenters):
    roidiff = roiactivity.diff()
    roipeaks = findpeaks(roidiff,thresh)
    data = [(peak,i,roicenters[i][1],roicenters[i][0])
            for i,step in enumerate(roipeaks)
            for peak in step]
    data = np.array(data)
    data = data[np.argsort(data[:,0]),:]
    return data
     
def steptimes(activity,thresh=1500):
    stepactivity = activity.iloc[:,17:25]
    data = roiactivations(stepactivity,thresh,stepcenter_cm)
    index = pd.Series(data[:,0],name='time')
    return pd.DataFrame(data[:,1:],
                        index=index,
                        columns=['stepindex',
                                 'stepcenterx',
                                 'stepcentery'])
                                 
def sliptimes(activity,thresh=1500):
    gapactivity = activity.iloc[:,25:32]
    data = roiactivations(gapactivity,thresh,slipcenter_cm)
    index = pd.Series(data[:,0],name='time')
    return pd.DataFrame(data[:,1:],
                        index=index,
                        columns=['gapindex',
                                 'gapcenterx',
                                 'gapcentery'])

def spatialaverage(activity,crossings,selector=lambda x:x.yhead):
    ypoints = []
    xpoints = np.linspace(rail_start_cm,rail_stop_cm,100)
    for s,side in crossings[['timeslice','side']].values:
        trial = activity.xs(s,level='time')
        xhead = trial.xhead
        yhead = selector(trial)
        if side == 'leftwards':
            xhead = max_width_cm - xhead
        curve = interp1d(xhead,yhead,bounds_error=False)
        ypoints.append(curve(xpoints))
    ypoints = np.array(ypoints)
    return xpoints,np.mean(ypoints,axis=0),stats.sem(ypoints,axis=0)
    
#def stepframeindices(activity,crossings,leftstep,rightstep):
#    indices = []
#    side = []
#    for index,trial in crossings.iterrows():
#        leftwards = trial.side == 'leftwards'
#        stepindex = leftstep if leftwards else rightstep
#        stepactivity = activity.xs(trial.timeslice,level='time',
#                                   drop_level=False).iloc[:,17:25]
#        stepdiff = stepactivity.diff()
#        steppeaks = findpeaks(stepdiff,1500)[stepindex]
#        steppeaks = [peak for peak in steppeaks
#                     if (activity.xhead[peak] > stepcenter_cm[rightstep] if leftwards else
#                         activity.xhead[peak] < stepcenter_cm[leftstep]).any()]
#        if len(steppeaks) > 0:
#            frameindex = min([activity.index.get_loc(peak) for peak in steppeaks])
#            indices.append(frameindex)
#            side.append(trial.side)
#    return indices,side

def getroipeaks(activity,roislice,trial,leftroi,rightroi,roicenters):
    leftwards = trial.side == 'leftwards'
    roiindex = leftroi if leftwards else rightroi
    roiactivity = activity.xs(trial.timeslice,level='time',
                              drop_level=False).ix[:,roislice]
    roidiff = roiactivity.diff()
    roipeaks = findpeaks(roidiff,1500)[roiindex]
    if len(roipeaks) > 0:
        print len(roipeaks)
    roipeaks = [peak for peak in roipeaks
                 if (activity.xhead[peak] > roicenters[rightroi] if leftwards else
                     activity.xhead[peak] < roicenters[leftroi]).any()]
    return roipeaks
    
def getsteppeaks(activity,trial,leftstep,rightstep):
    return getroipeaks(activity,slice(17,25),trial,leftstep,rightstep,stepcenter_cm)
    
def getslippeaks(activity,trial,leftgap,rightgap):
    return getroipeaks(activity,slice(25,32),trial,leftgap,rightgap,slipcenter_cm)

def roicrossings(activity,crossings,leftroi,rightroi,getpeaks):
    indices = []
    
    for index,trial in crossings.iterrows():
        roipeaks = getpeaks(activity,trial,leftroi,rightroi)
        if len(roipeaks) > 0:
            indices.append(index)
    return crossings.loc[indices]

def stepcrossings(activity,crossings,leftstep,rightstep):
    return roicrossings(activity,crossings,leftstep,rightstep,getsteppeaks)
    
def slipcrossings(activity,crossings,leftgap,rightgap):
    return roicrossings(activity,crossings,leftgap,rightgap,getslippeaks)

def roiframeindices(activity,crossings,leftroi,rightroi,getpeaks):
    indices = []
    side = []
    
    for index,trial in crossings.iterrows():
        roipeaks = getpeaks(activity,trial,leftroi,rightroi)
        if len(roipeaks) > 0:
            frameindex = min([activity.index.get_loc(peak) for peak in roipeaks])
            indices.append(frameindex)
            side.append(trial.side)
    print "done!"
    return indices,side
    
def stepframeindices(activity,crossings,leftstep,rightstep):
    return roiframeindices(activity,crossings,leftstep,rightstep,getsteppeaks)
    
def slipframeindices(activity,crossings,leftgap,rightgap):
    return roiframeindices(activity,crossings,leftgap,rightgap,getslippeaks)
    
def stepfeature(activity,crossings,leftstep,rightstep):
    indices,side = stepframeindices(activity,crossings,leftstep,rightstep)
    features = activity.ix[indices,:]
    features['side'] = side
    return features
#    side = pd.DataFrame(side,columns=['side'])
#    side.index = features.index
#    return pd.concat((features,side),axis=1)

def croproi(frame,roiindex,roicenter_pixels,cropsize=(300,300),background=None,
            flip=False,cropoffset=(0,0)):
    roicenter = roicenter_pixels[roiindex]
    roicenter = (roicenter[0] + cropoffset[0], roicenter[1] + cropoffset[1])
    
    frame = imgproc.croprect(roicenter,cropsize,frame)
    if background is not None:
        background = imgproc.croprect(roicenter,cropsize,background)
        frame = cv2.subtract(frame,background)
    if flip:
        frame = cv2.flip(frame,1)
    return frame
    
def cropstep(frame,stepindex,cropsize=(300,300),background=None,flip=False):
    return croproi(frame,stepindex,stepcenter_pixels,cropsize,background,flip)
    
def cropslip(frame,gapindex,cropsize=(300,300),background=None,flip=False):
    return croproi(frame,gapindex,slipcenter_pixels,cropsize,background,flip,
                   cropoffset=(-100,0))

def roiframes(activity,crossings,info,leftroi,rightroi,roiframeindices,croproi,
               cropsize=(300,300),subtractBackground=False):
    # Tile step frames    
    vidpaths = activitymovies.getmoviepath(info)
    timepaths = activitymovies.gettimepath(info)
    backpaths = activitymovies.getbackgroundpath(info)
    videos = [video.video(path,timepath) for path,timepath in zip(vidpaths,timepaths)]
    
    frames = []
    indices,side = roiframeindices(activity,crossings,leftroi,rightroi)
    for frameindex,side in zip(indices,side):
        leftwards = side == 'leftwards'
        roiindex = leftroi if leftwards else rightroi
        
        frame = videos[0].frame(frameindex)
        background = None
        if subtractBackground:
            timestamp = videos[0].timestamps[frameindex]
            background = activitymovies.getbackground(backpaths[0],timestamp)
        frame = croproi(frame,roiindex,cropsize,background,roiindex == rightroi)
        frames.append(frame)
    return frames
    
def stepframes(activity,crossings,info,leftstep,rightstep,
               cropsize=(300,300),subtractBackground=False):
    return roiframes(activity,crossings,info,leftstep,rightstep,
                     stepframeindices,cropstep,cropsize,subtractBackground)
                     
def slipframes(activity,crossings,info,leftgap,rightgap,
               cropsize=(300,300),subtractBackground=False):
    return roiframes(activity,crossings,info,leftgap,rightgap,
                     slipframeindices,cropslip,cropsize,subtractBackground)

def cropcrossings(x,slices,crop):
    def test_slice(s):
        return (x[s] > crop[0]) & (x[s] < crop[1])
    
    def crop_slice(s):
        valid_indices = np.nonzero(test_slice(s))[0]
        min_index = np.min(valid_indices)
        max_index = np.max(valid_indices)
        return slice(s.start+min_index,s.start+max_index+1)
    return [crop_slice(s) for s in slices if np.any(test_slice(s))]

def crossings(activity,midcross=True,crop=True):
    # Generate trajectories and crossings
    center = max_width_cm / 2.0
    cropleft = rail_start_pixels * width_pixel_to_cm
    cropright = rail_stop_pixels * width_pixel_to_cm
    xhead = activity.xhead
    crossings = np.ma.clump_unmasked(np.ma.masked_invalid(activity.xhead))
    if midcross:
        crossings = [s for s in crossings
        if xhead[s.start] > center and xhead[s.stop-1] < center
        or xhead[s.start] < center and xhead[s.stop-1] > center]
    if crop:
        crossings = cropcrossings(xhead,crossings,[cropleft,cropright])
        
    # Trial info
    trialinfo = pd.DataFrame([activity.iloc[s.start,1:8] for s in crossings])
    trialinfo.reset_index(inplace=True,drop=True)
    
    # Generate crossing features
    time = activity.index
    timeslice = pd.DataFrame([slice(time[s.start],time[s.stop-1])
                             for s in crossings],columns=['timeslice'])
    label = pd.DataFrame(['valid' for s in crossings],columns=['label'])
    position = pd.DataFrame([(activity.xhead[s].min(),activity.xhead[s].max())
                            for s in crossings],
                            columns=['xhead_min','xhead_max'])
    height = pd.DataFrame([activity.yhead[s].describe() for s in crossings])
    height.columns = 'yhead_' + height.columns
    height.columns = [c.replace('%','') for c in height.columns]
    speed = pd.DataFrame([activity.xhead_speed[s].abs().describe()
                         for s in crossings])
    speed.columns = 'xhead_speed_' + speed.columns
    speed.columns = [c.replace('%','') for c in speed.columns]
    duration = pd.DataFrame([(time[s.stop-1]-time[s.start]).total_seconds()
                            for s in crossings],
                            columns=['duration'])
    side = pd.DataFrame(['rightwards' if activity.xhead[s.start] < center else 'leftwards'
                        for s in crossings], columns=['side'])
    
    # Slowdown
    xspeed = activity.xhead_speed
    entrydistance = (cropright - cropleft) / 3.0    
    entrypoints = [xhead[s] < (cropleft + entrydistance)
    if xhead[s.stop-1] > xhead[s.start]
    else xhead[s] > (cropright - entrydistance)
    for s in crossings]
    exitpoints = [xhead[s] > (cropright - entrydistance)
    if xhead[s.stop-1] > xhead[s.start]
    else xhead[s] < (cropleft + entrydistance)
    for s in crossings]
        
    entryspeed = pd.DataFrame([np.abs(xspeed[s][v].mean())
    for s,v in zip(crossings,entrypoints)],columns=['entryspeed'])
    crossingspeed = pd.DataFrame([np.abs(xspeed[s][~v & ~x].mean())
    for s,v,x in zip(crossings,entrypoints,exitpoints)],
    columns=['crossingspeed'])    
    exitspeed = pd.DataFrame([np.abs(xspeed[s][v].mean())
    for s,v in zip(crossings,exitpoints)],columns=['exitspeed'])
    
    crossings = pd.DataFrame(crossings,columns=['slices'])
    return pd.concat([crossings,
                      timeslice,
                      label,
                      trialinfo,
                      duration,
                      position,
                      height,
                      speed,
                      side,
                      entryspeed,
                      crossingspeed,
                      exitspeed],
                      axis=1)
    