

import pyqtgraph as pg
from pyqtgraph import functions as fn


class CustomPlotDataItem(pg.PlotDataItem):
    def __init__(self, *args, **kargs):
        pg.PlotDataItem.__init__(self, *args, **kargs)

    def setSymbolBrush(self, brush=None, *args, **kargs):
        if brush is None:
            brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        self.updateItems()

