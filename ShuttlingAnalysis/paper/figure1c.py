# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 20:00:50 2014

@author: IntelligentSystems
"""

import stats
import sessions
import trajectories
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lesion_order = np.array([0, 4, 12, 10, 2, 18, 8, 14, 20, 16, 6])
control_order = lesion_order+1

class figure1c:
    def __init__(self, lesions, controls):
        self.lesions = lesions
        self.controls = controls
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.axes()
        
        def gettrialstats(subjects):
            return [np.array([stats.meanstd([np.mean(t,axis=0)[1]
            for t in s.tolist()[slice(1,None)]])
            for s in a])
            for a in subjects]
            
        def getstatsummary(stats):
            meantimes = [[a[i,0] for a in stats] for i in range(3)]
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
            plt.errorbar(x,mean,error,fmt=None,color=color,ecolor='k',linewidth=2,capthick=2,markersize=0)
        
        def drawseparators(ax,xticks):
            ylims = plt.ylim()
            for i in range(len(xticks)-1):
                separatorxi = (xticks[i]+xticks[i+1])/2
                ax.plot((separatorxi,separatorxi),ylims,'k--')
            plt.ylim(ylims)
        
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
        xticks = [scale*i+len(controltimes)-0.5 for i in range(3)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(['stable','partial','unstable'])
        drawseparators(ax,xticks)
        plt.ylabel('nose height (cm)')
        
            
def genfromtxt(folders):
    lesionfolders = [folders[i] for i in lesion_order[1:]]
    controlfolders = [folders[i] for i in control_order]
                    
    print("Processing lesions...")
    lesions = sessions.genfromsubjects(lesionfolders,[4,10,-1],trajectories)
    print("Processing controls...")
    controls = sessions.genfromsubjects(controlfolders,[4,10,-1],trajectories)
    return figure1c(lesions, controls)