# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:09:56 2014

@author: Gonçalo
"""

import numpy as np
import scipy.signal as signal
from bisect import bisect_left

def loadts(path,dtype=np.uint16,nchannels=1):
    data = np.memmap(path,dtype,mode='c')
    nsamples = len(data) / nchannels
    return np.reshape(data,(nsamples,nchannels))
    
def bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')
    
def bandpassfilter(data,lowcut,highcut,fs,order=6):
    b,a = bandpass(lowcut,highcut,fs,order=6)
    return signal.filtfilt(b,a,data,axis=0)
    
def downsample10x(ts):
    return signal.decimate(ts,10,axis=0)
    
def downsampleAdc(adc):
    ds = downsample10x(adc)
    dss = downsample10x(ds)
    return dss.astype(np.float32)
    
def preprocessAdc(path,targetpath):
    adc = loadts(path,nchannels=8)
    ds = downsampleAdc(adc)
    del adc
    ds.tofile(targetpath)
    return ds
    
def findpeaksMax(ts,thresh,axis=-1):
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
    
def findpeaks(ts,thresh,mindistance=0):
    thresholded = ts > thresh if thresh > 0 else ts < thresh
    crossings = np.insert(np.diff(np.int8(thresholded),axis=0) > 0,0,False)
    indices = np.nonzero(crossings)[0]
    if mindistance > 0:
        ici = np.insert(np.diff(indices),0,0)
        return indices[ici >= mindistance]
    return indices
    
def loadwaves(path,nsamples,dtype=np.float32):
    data = np.memmap(path,dtype)
    nwaves = len(data) / (nsamples + 2)
    return np.reshape(data,(nwaves,nsamples + 2))
    
def offsetdata(data,offset,axis=-1):
    return data + (np.arange(data.shape[axis]) * offset)
    
def aligndata(data,evts,before,after):
    return np.array([data[evt-before:evt+after,:] for evt in evts])
    
def alignspks(spks,evts,before,after):
    return [np.array([s-evt for s in spks if s-evt >= before and s-evt <= after]) for evt in evts]
    
import princomp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collectionselector
from matplotlib.widgets import Lasso, LassoSelector
from matplotlib import path

def allspikes(spks):
    fig = plt.figure()
    for i in range(32):
        sc = np.nonzero(spks[:,0] == i)[0]
        ax = fig.add_subplot(4,8,i + 1)
        if len(sc) == 0:
            continue
        
        spksdata = spks[sc,2:]
        averagetrace(spksdata,1,drawmean=False,ax=ax,color='g')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
def allspikepcs(spks,labels,nx=4,ny=8):
    fig = plt.figure()
    nchannels = nx * ny
    for i in range(nchannels):
        sc = np.nonzero(spks[:,0] == i)[0]
        ax = fig.add_subplot(nx,ny,i + 1)
        if len(sc) == 0:
            continue
        
        spksdata = spks[sc,2:]
        coeff,score,latent = princomp.princomp(spksdata)
        fx = score[0,:]; fxl = ''
        fy = score[1,:]; fyl = ''
        if i < len(labels):
            spklabels = labels[i]
        else:
            spklabels = np.zeros(np.shape(spksdata)[0])
        labels.append(spklabels)
        sortspikes(fx,fy,fxl,fyl,spksdata,spklabels,ax=ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

def averagetrace(traces,alpha=0.1,Fs=1,drawmean=True,ax=None,**kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    time = maketimeaxis(np.shape(traces)[1],Fs)
    ax.plot(time,traces.T,alpha=alpha,**kwargs)
    if drawmean:
        mu = np.mean(traces,0)
        ax.plot(time,mu,'r',linewidth=3)
    
def plotraster(rsts,Fs=1):
    plt.figure()
    [plt.plot(rsts[i]*1./Fs,i*np.ones(np.shape(rsts[i])),'k.') for i,r in enumerate(rsts)]
    
def manualfilter(data):
    plt.plot(data.T,'k',alpha=0.1)
    indices = []
    for i,d in enumerate(data):
        plt.plot(d,'r')
        if plt.waitforbuttonpress():
            plt.plot(d,'g')
            indices.append(i)
    return indices
    
def maketimeaxis(nsamples,Fs,offset=0):
    return (np.arange(nsamples) + offset) * 1. / Fs
    
def timetranslate(time,timebasissync,translatesync,Fs1=1,Fs2=1):
    timesync = bisect_left(timebasissync,time)
    timesyncdiff = (time - timebasissync[timesync]) / Fs1
    return np.int(translatesync[timesync] + timesyncdiff * Fs2)
    
# FEATURE EXTRACTION
def peaktopeakamplitude(waves):
    return np.max(waves,1) - np.min(waves,1)
    
def energy(waves):
    return np.sum(np.power(np.abs(waves),2),1)    
    
def windowdiscriminator(waves,xmin,xmax,ymin,ymax):
    yhits = (waves >= ymin) & (waves < ymax)
    xmask = np.zeros(np.shape(waves)[1]).astype(np.bool)
    xmask[xmin:xmax] = True
    hits = yhits & xmask
    return np.sum(hits,1)
    
# SPIKE SORTING
def sortspikes3(f1,f2,f1l,f2l,spks,labels):
    nlabels = 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cmap = plt.cm.Accent;
    cmaplut = np.linspace(0,1,nlabels)
    colors = np.array([cmap(cmaplut[l]) for l in labels])
    
    pts = ax.scatter3D(f1,f2,facecolors=colors,edgecolors='none')
    
def sortspikes(f1,f2,f1l,f2l,spks,labels,ax=None):
    nlabels = 10
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    fig = ax.figure
    
    cmap = plt.cm.Accent;
    cmaplut = np.linspace(0,1,nlabels)
    colors = np.array([cmap(cmaplut[l]) for l in labels])
    
    pts = ax.scatter(f1,f2,edgecolors='none')
    ax.set_xlabel(f1l)
    ax.set_ylabel(f2l)
    vax = [None]
        
    def onkeypress(evt):
        if evt.inaxes != ax:
            return
        
        if vax[0] is not None and not plt.fignum_exists(vax[0].figure.number):
            vax[0] = None
            
        if evt.key == 'a':
            selector.selectall()
            
        if evt.key == 'z':
            selector.reset()
            
        if evt.key == 'v':
            if len(selector.ind) > 0:
                ss = spks[selector.ind]
                sl = labels[selector.ind]
            else:
                ss = spks
                sl = labels
                
            for i in range(nlabels):
                cs = ss[sl == i]
                if len(cs) > 0:
                    if vax[0] == None:
                        vfig = plt.figure()
                        vax[0] = vfig.add_subplot(111)
                    plt.figure(vax[0].figure.number)
                    averagetrace(cs,alpha=1,drawmean=False,color=cmap(cmaplut[i]),ax=vax[0])
                    plt.draw()
        
        if len(selector.ind) > 0:
            if evt.key == 'c':
                if vax[0] is not None:
                    vax[0].clear()
                    plt.figure(vax[0].figure.number)
                    vax[0].figure.canvas.draw_idle()
            
            if evt.key.isdigit():
                label = int(evt.key)
                labels[selector.ind] = label
                colors = np.array([cmap(cmaplut[l]) for l in labels])
                selector.fc = colors
    
    fig.canvas.mpl_connect('key_press_event',onkeypress)
    selector = collectionselector.CollectionSelector(ax,pts,color_other='r')
    return selector
    #setpicker(f1,f2)
    
# POINT PICKER
def setspikepicker(x,y,spks):
    ax = []
    def callback(ind,altpicker):
        if len(ax) > 0:
            if not plt.fignum_exists(ax[0].figure.number):
                ax.pop()
            else:
                if not altpicker:
                    ax[0].clear()
                plt.figure(ax[0].figure.number)
                
        if len(ax) == 0:
            fig = plt.figure()
            ax0 = fig.add_subplot(111)
            ax.append(ax0)
        
        if len(ind) > 0:
            alpha = 0.1 if altpicker else 1
            drawmean = False if altpicker else True
            averagetrace(spks[ind],alpha=alpha,drawmean=drawmean,ax=ax[0],color='k')
        plt.title('N = ' + str(len(ind)))
        plt.draw()
    setpicker(x,y,callback)
    
def setpicker(x,y,callback=None):
    fig = plt.gcf()
    lassos = []
    selected = []
    xys = zip(x,y)
    altpicker = [False]
    def clearlassos():
        while len(lassos) > 0:
            lasso = lassos.pop()
            fig.canvas.widgetlock.release(lasso)
            del lasso
        fig.canvas.draw_idle()
    
    def onlasso(verts):
        print 'onlasso'
        p = path.Path(verts)
        contained = p.contains_points(xys)
        ind = [i for i,c in enumerate(contained) if c]
        selected.append(plt.plot(x[ind],y[ind],'r.'))
        clearlassos()
        
        if callback is not None:
            callback(ind,altpicker[0])
            
    def onkeypress(evt):
        print evt.key
    
    def onbuttonrelease(evt):
        print evt
        if evt.inaxes is None:
            return
        clearlassos()
    
    def onbuttonpress(evt):
        if fig.canvas.widgetlock.locked():
            return
        if evt.inaxes is None:
            return
            
        while len(selected) > 0:
            lines = selected.pop()
            for line in lines:
                evt.inaxes.lines.remove(line)
        
        altpicker[0] = evt.key == 'control'            
        lasso = Lasso(evt.inaxes, (evt.xdata, evt.ydata), onlasso)
        lassos.append(lasso)
        fig.canvas.widgetlock(lasso)
    
    #fig.canvas.mpl_connect('key_press_event',onkeypress)
    fig.canvas.mpl_connect('button_press_event',onbuttonpress)
    fig.canvas.mpl_connect('button_release_event',onbuttonrelease)