# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:40:15 2013

@author: gonca_000
"""

import stats
import sessions
import shuttling
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lesion_order = np.array([0, 4, 12, 10, 2, 18, 8, 14, 20, 16, 6])
control_order = lesion_order+1

class figure1b:
    def __init__(self, lesions, controls):
        self.lesions = lesions
        self.controls = controls
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.axes()
        
        def gettrialstats(subjects):
            return [np.array([stats.meanstd([t.total_seconds() for t in np.diff(s.rewards)]) for s in a]) for a in subjects]
            
        def getstatsummary(stats):
            meantimes = [[a[i,0] for a in stats] for i in range(4)]
            mean = [np.mean(d) for d in meantimes]
            std = [np.std(d) for d in meantimes]
            return mean,std
            
        def plottrialstats(ax,data,label=None,offset=0,scale=1):
            mean = data[:,0]
            std = data[:,1]
            ax.xaxis.set_major_locator(ticker.IndexLocator(1,0))
            x = (np.arange(len(data))*scale)+offset
            (_,caps,_) = plt.errorbar(x,mean,std,fmt=None,label=label,zorder=100,capthick=1,alpha=0.4)
            for cap in caps:
                cap.remove()
            plt.xlim(-0.5-offset/2,(len(data)*scale)-0.5)
            
        def plotstatsummary(ax,data,color,offset=0):
            mean,error = getstatsummary(data)
            x = (np.arange(len(mean))*scale)+(len(data)/2)+offset
            plt.errorbar(x,mean,error,fmt='--',color=color,ecolor='k',linewidth=2,capthick=2,markersize=0)    
        
        scale = 46
        lesiontimes = gettrialstats(self.lesions)
        controltimes = gettrialstats(self.controls)
        
        ax.set_color_cycle(['b'])
        [plottrialstats(ax,data,'control',i,scale) for i,data in enumerate(controltimes)]
        plotstatsummary(ax,controltimes,'b')
        
        ax.set_color_cycle(['r'])
        [plottrialstats(ax,data,'lesion',i+len(controltimes),scale) for i,data in enumerate(lesiontimes)]
        plotstatsummary(ax,lesiontimes,'r',len(controltimes))
        
        handles, labels = ax.get_legend_handles_labels()
        plt.legend((handles[0],handles[len(controltimes)]),(labels[0],labels[len(controltimes)]))
        ax.set_xticks([scale*i+len(controltimes)-0.5 for i in range(4)])
        ax.set_xticklabels([1,2,3,4])
        plt.xlabel('sessions')
        plt.ylabel('time between rewards (s)')
        
            
def genfromtxt(folders):
    lesionfolders = [folders[i] for i in lesion_order]
    controlfolders = [folders[i] for i in control_order]
                    
    print "Processing lesions..."
    lesions = sessions.genfromsubjects(lesionfolders,range(1,5),shuttling)
    print "Processing controls..."
    controls = sessions.genfromsubjects(controlfolders,range(1,5),shuttling)
    return figure1b(lesions, controls)