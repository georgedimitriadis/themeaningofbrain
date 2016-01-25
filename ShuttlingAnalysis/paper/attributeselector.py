# -*- coding: utf-8 -*-
"""
Created on Wed Sep 03 10:43:10 2014

@author: Gonçalo
"""

import numpy as np
from matplotlib.colors import colorConverter
from matplotlib.widgets import CheckButtons
from matplotlib.path import Path

class AttributeSelector(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """
    def __init__(self, ax, collection, rax, raxmap, labels, labelcolors):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.labelcolors = [colorConverter.to_rgba(c) for c in labelcolors]
        self.labels = labels

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)
        self.sfc = self.fc.copy()

        self.state = [False] * len(labels)
        self.check = CheckButtons(rax,labels,self.state)
        self.check.on_clicked(self.onclick)
        self.raxmap = raxmap
        self.ind = []
        
    def updateselection(self, color):
        if len(self.ind) > 0:
            self.sfc[self.ind, :] = color
        self.collection.set_facecolors(self.sfc)
        self.canvas.draw_idle()

    def onclick(self, label):
        self.ind = raxmap[label]
        labelindex = self.labels.index(label)
        self.state[labelindex] = not self.state[labelindex]
        if self.state[labelindex]:
            color = self.labelcolors[labelindex]
        else:
            color = self.fc[self.ind,:]
        
        self.updateselection(color)

    def disconnect(self):
        self.check.disconnect_events()
        #self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()
    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], c='g', s=80)
    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    raxmap = {'l1':[0,2,3],'l2':[6,3,7,8,28],'l3':[1,6,7,2,6]}
    selector = AttributeSelector(ax, pts, rax, raxmap, ['l1', 'l2', 'l3'], ['r','g','b'])

    plt.draw()