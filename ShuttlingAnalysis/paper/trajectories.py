# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:12:45 2013

@author: gonca_000
"""

import os
import numpy as np

max_height_cm = 24.0
height_pixel_to_cm = max_height_cm / 680.0
width_pixel_to_cm = 50.0 / 1280.0
rail_height_pixels = 100
frames_per_second = 120.0

# stores tip trajectories from shuttling task
# * data rows are frames
# * data cols are [xleft,yleft,xright,yright]

class trajectories:
    def __init__(self, data, slices=None, **kwargs):
        self.data = data
        if slices is None:
            mask = kwargs.get('mask')
            if mask is None:
                mask = np.ma.masked_equal(data[:,0],-1)
            slices = np.ma.clump_unmasked(mask)
        self.slices = slices
        
    def __repr__(self):
        if self.slices is None:
            return str.format("trajectories({0})", self.data)
        else:
            return str.format("trajectories({0}, {1})", self.data, self.slices)
        
    def tolist(self):
        return [self.data[s,:] for s in self.slices]
        
def concatenate(tss):
    data = np.concatenate([ts.data for ts in tss],axis=0)
    offset = 0
    slices = []
    for ts in tss:
        for s in ts.slices:
            slices.append(slice(s.start+offset,s.stop+offset,s.step))
        offset += ts.data.shape[0]
    return trajectories(data,np.array(slices))
        
def scale(ts,
          sx=width_pixel_to_cm,
          sy=height_pixel_to_cm,
          by=rail_height_pixels,
          my=max_height_cm):
    scaled = [0,my,0,my] - (ts.data + [0,by,0,by]) * [-sx,sy,-sx,sy]
    return trajectories(scaled,ts.slices)
    
def crossings(ts,center=640):
    return trajectories(ts.data,np.array([s for s in ts.slices if
    (ts.data[s.start,0] < center and ts.data[s.stop-1,0] > center) or
    (ts.data[s.start,0] > center and ts.data[s.stop-1,0] < center)]))
    
def lengthfilter(ts,minlength=None,maxlength=None):
    return trajectories(ts.data,np.array([s for s in ts.slices if
    s.stop-s.start >= minlength and
    (maxlength is None or s.stop-s.start <= maxlength)]))
    
def heightfilter(ts,minheight=None,maxheight=None):
    return trajectories(ts.data,np.array([s for s in ts.slices if
    min(ts.data[s,1]) > minheight and max(ts.data[s,1]) < maxheight]))
    
def speed(ts,time):
    timedelta = np.diff(time)
    deltaT = np.tile(timedelta,(ts.data.shape[1],1)).T
    speed = np.insert(np.diff(ts.data,axis=0) / deltaT,0,0,axis=0)
    return trajectories(speed,ts.slices)
    
def speedbins(ts,sp,bins=100):
    def computebins(s):
        weights = sp.data[s,0] if s.start < s.stop else -sp.data[s,0]
        binsums = np.histogram(ts.data[s,0],bins,weights=weights)[0]
        bincounts = np.histogram(ts.data[s,0],bins,weights=None)[0]
        return binsums / bincounts
    return [computebins(s) for s in ts.slices]
    
def crop(ts,crop=[200,1000]):
    def test_slice(s):
        return (ts.data[s,0] > crop[0]) & (ts.data[s,0] < crop[1])
    
    def crop_slice(s):
        valid_indices = np.nonzero(test_slice(s))[0]
        min_index = np.min(valid_indices)
        max_index = np.max(valid_indices)
        return slice(s.start+min_index,s.start+max_index+1)
    return trajectories(ts.data,np.array([crop_slice(s) for s in ts.slices
    if np.any(test_slice(s))]))

def mirrorleft(ts):
    return trajectories(ts.data,np.array([slice(s.stop,s.start,-1)
    if ts.data[s.start,0] > ts.data[s.stop,0] else s
    for s in ts.slices]))
        
def samedirection(ts):
    def mirrortrial(s):
        result = ts.data[s,:]
        if ts.data[s.start,0] > ts.data[s.stop,0]:
            x = result[:,0]
            result = result.copy()
            result[:,0] = -x + np.min(x) + np.max(x)
        return result    
    return [mirrortrial(s) for s in ts.slices]
    
def samedirectionspeed(ts,sp):
    def mirrortrial(s):
        result = sp.data[s,:]
        if ts.data[s.start,0] > ts.data[s.stop,0]:
            xsp = result[:,0]
            result = result.copy()
            result[:,0] = -xsp
        return result
    return [mirrortrial(s) for s in ts.slices]
        
def left(ts):
    return trajectories(ts.data,np.array([s for s in ts.slices
    if ts.data[s.start,0] > ts.data[s.stop,0]]))
        
def right(ts):
    return trajectories(ts.data,np.array([s for s in ts.slices
    if ts.data[s.start,0] < ts.data[s.stop,0]]))
        
def crossindices(ts,center=25.0):
    return np.array([next(i+s.start for i,x in enumerate(ts.data[s,0]) if
    x < center) for s in ts.slices[1:]])

def genfromtxt(path):
    trajectoriespath = os.path.join(path, 'Analysis/trajectories.csv')
    data = np.genfromtxt(trajectoriespath)
    return crossings(trajectories(data))
    #return scale(crop(crossings(trajectories(data))))
    
## FEATURES ##
# minx, maxx, miny, maxy, length, meany, stdy, meanx, stdx
def features(ts,funs):
    if np.iterable(ts):
        return np.array([[f(t[:,i]) for i,f in funs] for t in ts])
    else:
        return np.array([[f(ts.data[s,i]) for i,f in funs] for s in ts.slices])