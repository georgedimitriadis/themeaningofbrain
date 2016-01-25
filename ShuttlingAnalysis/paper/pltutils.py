# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 14:09:10 2013

@author: IntelligentSystems
"""

import numpy as np
import collectionselector
import scipy.stats as scistats
import matplotlib.pyplot as plt

def regressionline(xi,yi,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    w0,w1,r,p,err = scistats.linregress(xi,yi)
    if np.isnan(w0):
        return
        
    line = w0*xi + w1
    kwargs['marker'] = None
    r_string = r'$r_s=%.2f,$' % r**2
    p_string = r'$p < 0.01$' if p < 0.01 else r'$p = %.2f$' % p
    ax.text(max(xi), max(line), r_string+'\n'+p_string,verticalalignment='center')
    return ax.plot(xi,line,**kwargs)

def hbracket(x,y,width,label=None,tickheight=1,color='k'):
    ax = plt.gca()
    ax.annotate(label,
            xy=(x, y), xycoords='data',
            xytext=(x, y+tickheight), textcoords='data',
            horizontalalignment='center',
            arrowprops=dict(arrowstyle="-[,widthB="+str(width), #linestyle="dashed",
                            color=color,
                            patchB=None,
                            shrinkB=0
                            ),
            )

def click_data_action(figure,ondataclick):
    def onclick(event):
        if event.button == 3 and event.xdata is not None and event.ydata is not None:
            ondataclick(event)
    figure.canvas.mpl_connect('button_press_event',onclick)
    
def fix_font_size():
    ax = plt.gca()
    plt.tight_layout()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
def featuretiles(data,plotter,labels=None,facecolors='b'):
    fig = plt.figure()
    ndims = data.shape[1]
    selectors = []
    
    def onplot(ind):
        plt.sca(viewax)
        viewax.clear()
        plotter(ind)
            
    def onselect(ind):
        for sel in selectors:
            sel.ind = ind
            sel.updateselection()
        onplot(ind)

    halfdims = int(ndims / 2)
    rowhalf = halfdims + ndims % 2
    viewax = plt.subplot2grid((ndims,ndims),(rowhalf,0),rowspan=halfdims,colspan=halfdims)
    
    for i in range(ndims):
        for j in range(i,ndims):
            ax = fig.add_subplot(ndims,ndims,i*ndims+j+1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if i == j:
                ax.plot(data[:,i])
                if labels is not None:
                    ax.set_xlabel(labels[i])
            else:
                selectors.append(featureview(data[:,i],data[:,j],None,ax,onselect,facecolors))
                #ax.plot(data[:,i],data[:,j],'.')
        
def featureview(f1,f2,plotter=None,ax=None,onselection=None,facecolors='b'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    pts = ax.scatter(f1,f2,facecolors=facecolors,edgecolors='none')

    def onkeypress(evt):
        if plotter is None:
            return
        if evt.key == 'v':
            ind = selector.ind
            if len(ind) > 0:
                plotter(ind)
    
    ax.figure.canvas.mpl_connect('key_press_event',onkeypress)
    selector = collectionselector.CollectionSelector(ax,pts,color_other='b',onselection=onselection)
    return selector