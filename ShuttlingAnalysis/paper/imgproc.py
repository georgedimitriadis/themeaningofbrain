# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:47:07 2014

@author: IntelligentSystem
"""

import cv2
import numpy as np

def boundingbox(points):
    minx = None
    maxx = None
    miny = None
    maxy = None
    for x,y in points:
        if minx is None:
            minx = x
            maxx = x
            miny = y
            maxy = y
        else:
            minx = min(minx,x)
            maxx = max(maxx,x)
            miny = min(miny,y)
            maxy = max(maxy,y)
    return (minx,maxx),(miny,maxy)
    
def centroid(points):
    cx = 0
    cy = 0
    for x,y in points:
        cx += x
        cy += y
    cx /= len(points)
    cy /= len(points)
    return cx,cy
    
def croprect(centroid,shape,frame):
    halfh = shape[0] / 2
    halfw = shape[1] / 2
    top = max(0,centroid[0] - halfh)
    bottom = min(frame.shape[0]-1,centroid[0] + halfh)
    left = max(0,centroid[1] - halfw)
    right = min(frame.shape[1]-1,centroid[1] + halfw)
    return frame[slice(top,bottom),slice(left,right)]
    
import video
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
    
def distancematrix(frames,normType=cv2.cv.CV_L2):
    result = np.zeros((len(frames),len(frames)))
    for i in xrange(len(frames)):
        for j in xrange(i,len(frames)):
            distance = cv2.norm(frames[i],frames[j],normType)
            result[i,j] = distance
            if i != j:
                result[j,i] = distance
    return result
    
def cluster(frames,vid=None,indices=None,labels=None):
    drawlabels = [False]
    if labels is None:
        labels = np.zeros(len(frames),dtype=int)
    fig = plt.figure()
    distance = distancematrix(frames,cv2.cv.CV_L1)
    Z = sch.linkage(distance,'complete')
    ax1 = fig.add_axes([0.05,0.1,0.4,0.6])
    ax2 = fig.add_axes([0.05,0.71,0.4,0.2])
    R = sch.dendrogram(Z)
    ax2.set_title('frame clusters')
    leaves = R['leaves']
    ax2.set_xticks([])
    ax2.set_yticks([])
    sframes = frames[leaves]
    distance = distance[leaves,:]
    distance = distance[:,leaves]
    axcolor = fig.add_axes([0.46,0.1,0.02,0.6])
    im = ax1.imshow(distance,aspect='auto')
    plt.colorbar(im, cax=axcolor)
    fn,fm = sframes[0].shape
    xmin,xmax = ax2.get_xlim()
    ax3 = plt.subplot2grid((3,2),(0,1), rowspan=3)
    def drawframes(ax,frameslice=slice(None)):
        pframes = sframes[frameslice]
        ntiles = int(np.ceil(np.sqrt(len(pframes))))
        if ntiles == 0:
            return
        tiles = tile(pframes,ntiles,ntiles)
        ax.clear()
        ax.imshow(tiles[0])
        if (labels is not None) and drawlabels[0]:
            leafslice = leaves[frameslice]
            plabels = labels[leafslice]
            xlabels = fm * (np.arange(len(pframes)) % ntiles) + 0.1 * fm
            ylabels = fn * (np.arange(len(pframes)) / ntiles) + 0.85 * fn
            [ax.text(x,y,l,color='r') for x,y,l in zip(xlabels,ylabels,plabels)]
        ax.set_xticks([])
        ax.set_yticks([])
    drawframes(ax3)
    ax3.set_title('sorted frames')
    
    def getframelim(ax):
        lmin,lmax = ax.get_xlim()
        lmin = len(leaves) * (lmin / xmax) - 0.5
        lmax = len(leaves) * (lmax / xmax) - 0.5
        return lmin,lmax
        
    def getframeslice(lmin,lmax):
        return slice(int(np.ceil(lmin)),int(np.floor(lmax))+1)
    
    def onlimitchanged(ax):
        lmin,lmax = getframelim(ax)
        frameslice = getframeslice(lmin,lmax)
        drawframes(ax3,frameslice)
        ax1.set_xlim(lmin,lmax)
        
    def onkeypress(evt):
        lmin,lmax = getframelim(ax2)
        frameslice = getframeslice(lmin,lmax)
        if evt.key == 'l':
            drawlabels[0] = not drawlabels[0]
        else:
            try:
                label = int(evt.key)
                labels[leaves[frameslice]] = label
            except ValueError:
                return
        
        drawframes(ax3,frameslice)
        fig.canvas.draw_idle()
    
    def onmouseclick(evt):
        if evt.inaxes == ax3:
            lmin,lmax = getframelim(ax2)
            frameslice = getframeslice(lmin,lmax)
            ntiles = int(np.ceil(np.sqrt(frameslice.stop-frameslice.start)))
            x = int(evt.xdata / fm)
            y = int(evt.ydata / fn)
            fi = y * ntiles + x
            if vid is not None and indices is not None:
                idx = indices[leaves[fi]]
                video.showmovie(vid,idx)
    h1 = fig.canvas.mpl_connect('button_press_event',onmouseclick)
    h2 = fig.canvas.mpl_connect('key_press_event',onkeypress)
    h3 = ax2.callbacks.connect('xlim_changed', onlimitchanged) 
    return Z, R, labels, (h1, h2, h3)

def tile(frames,width,height,labels=None):
    frameshape = np.shape(frames[0])
    frametype = frames[0].dtype

    pages = []
    pagecount = width * height
    npages = len(frames) / pagecount
    npages = npages + 1 if len(frames) % pagecount != 0 else npages
    for i in range(npages):
        page = np.zeros((frameshape[0] * height,frameshape[1] * width),frametype)
        for k in range(pagecount):
            fi = i * pagecount + k
            if fi >= len(frames):
                continue
            
            frame = frames[fi]
            if labels is not None:
                frame = frame.copy()
                cv2.putText(frame,labels[fi],(0,frame.shape[0]-1),
                            cv2.cv.CV_FONT_HERSHEY_COMPLEX,1,
                            (255,255,255,255))
            offsetH = int(k / width) * frameshape[0]
            offsetW = int(k % width) * frameshape[1]
            sliceH = slice(offsetH,offsetH + frameshape[0])
            sliceW = slice(offsetW,offsetW + frameshape[1])
            page[sliceH,sliceW] = frame
        pages.append(page)
    return pages
    
def average(frames,scale=1.0,shift=0.0):
    if len(frames) == 0:
        raise ValueError('frames is an empty sequence')
        
    result = frames[0].astype(np.float32) * scale + shift
    for i in range(1,len(frames)):
        result += frames[i] * scale + shift
    result /= len(frames)
    return result
    
def crop(frames,xslice,yslice):
    return [frame[yslice,xslice] for frame in frames]
    
def calcpca(data):
    U, s, Vt = np.linalg.svd(data,full_matrices=False)
    V = Vt.T
    
    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    V = V[:, ind]
    return U, s, V
    
def pca(X):
  # Principal Component Analysis
  # input: X, matrix with training data as flattened arrays in rows
  # return: projection matrix (with important dimensions first),
  # variance and mean

  #get dimensions
  num_data,dim = X.shape

  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      M = np.dot(X,X.T) #covariance matrix
      e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
      tmp = np.dot(X.T,EV).T #this is the compact trick
      V = tmp[::-1] #reverse since last eigenvectors are the ones we want
      S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
  else:
      U,S,V = np.linalg.svd(X)
      V = V[:num_data] #only makes sense to return the first num_data

  #return the projection matrix, the variance and the mean
  return V,S,mean_X