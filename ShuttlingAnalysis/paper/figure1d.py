# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 20:00:50 2014

@author: IntelligentSystems
"""

import stats
import sessions
import pltutils
import trajectories
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lesion_order = np.array([0, 4, 12, 10, 2, 18, 8, 14, 20, 16, 6])
control_order = lesion_order+1

class figure1d:
    def __init__(self, lesions, controls):
        self.lesions = lesions
        self.controls = controls
        
    def plot(self, legend=False, ax=None):
        #bracketmaxoffset = 0.1
        bracketmaxscale = 0.1
        bracketminoffset = 0.65
        brackettickheight = 0.5
        
        if ax is None:
            ax = plt.axes()
            
        def getminmaxrange(data):
            concat = np.concatenate(data)
            rangemin = np.min(concat[:,0]-concat[:,1])
            rangemax = np.max(np.sum(concat, axis=1))
            return rangemin,rangemax
        
        def gettrialstats(subjects):
            #return [np.array([stats.meanstd([np.mean(t,axis=0)[1]
            return [np.array([stats.meanstd([np.shape(t)[0] / 120.0
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
        
        scale = 46
        lesionstats = gettrialstats(self.lesions)
        controlstats = gettrialstats(self.controls)
        
        ax.set_color_cycle(['b'])
        [plottrialstats(ax,data,'control',i,scale) for i,data in enumerate(controlstats)]
        plotstatsummary(ax,controlstats,'b')
        
        ax.set_color_cycle(['r'])
        [plottrialstats(ax,data,'lesion',i+len(controlstats),scale) for i,data in enumerate(lesionstats)]
        plotstatsummary(ax,lesionstats,'r',len(controlstats))
        
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            plt.legend((handles[0],handles[len(controlstats)]),(labels[0],labels[len(controlstats)]))
        xticks = [scale*i+len(controlstats)-0.5 for i in range(3)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(['stable','partial','unstable'])

        nsessions = len(lesionstats[0])        
        minstdl,maxstdl = getminmaxrange(lesionstats)
        minstdc,maxstdc = getminmaxrange(controlstats)
        minstd = min(minstdl,minstdc)
        maxstd = max(maxstdl,maxstdc)
        lesionstats = [[a[i,0] for a in lesionstats] for i in range(nsessions)]
        controlstats = [[a[i,0] for a in controlstats] for i in range(nsessions)]
        
        ### SIGNIFICANCE (BETWEEN GROUPS) ###
        #significance = 0.05
        #for i in range(len(xticks)):
        #    sigtest = scipy.stats.ttest_ind(lesionstats[i],controlstats[i])[1]
        #    print sigtest,"groups"
        #    testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
        #    pltutils.hbracket(xticks[i],maxstd+maxstd*bracketmaxscale,2,label=testlabel,tickheight=brackettickheight)
            
        #################################
        
        ### SIGNIFICANCE (BETWEEN CONDITIONS) ###
        #significance = 0.05
        #sigtest = scipy.stats.ttest_ind(lesionstats[0]+controlstats[0],lesionstats[-1]+controlstats[-1])[1]
        #print sigtest,"conditions"
        #testlabel = str.format("*",sigtest) if sigtest < significance else 'n.s.'
        #pltutils.hbracket(xticks[1],minstd+minstd*bracketmaxscale,(xticks[-1]-xticks[0])/10,label=testlabel,tickheight=-1.5*brackettickheight)
        #####################################
        
        #ax.set_ylim([-1,7.5])
        breaks = np.diff(xticks)/2 + xticks[0:-1]
        ylims = ax.get_ylim()
        ax.vlines(breaks,ylims[0],ylims[1],'k','dashed')
        #plt.ylabel('nose height (cm)')
        plt.ylabel('time to cross obstacles (s)')
            
def genfromtxt(folders):
    lesionfolders = [folders[i] for i in lesion_order[1:]]
    controlfolders = [folders[i] for i in control_order]
                    
    print "Processing lesions..."
    lesions = sessions.genfromsubjects(lesionfolders,[4,10,-1],trajectories)
    print "Processing controls..."
    controls = sessions.genfromsubjects(controlfolders,[4,10,-1],trajectories)
    return figure1d(lesions, controls)