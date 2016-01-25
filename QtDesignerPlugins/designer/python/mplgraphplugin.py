from PyQt4 import QtGui, QtDesigner

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from mplgraphwidget import MplGraphWidget

class MplGraphPlugin(QtDesigner.QPyDesignerCustomWidgetPlugin):

    def __init__(self, parent = None):
        QtDesigner.QPyDesignerCustomWidgetPlugin.__init__(self)
        self.initialized = False

    def initialize(self, core):
        if self.initialized:
            return
        self.initialized = True

    def isInitialized(self):
        return self.initialized

    def createWidget(self, parent):
        fc = MplGraphWidget(parent,width=2, height=2)
        return fc


    def name(self):
        return "MplGraphWidget"

    def group(self):
        return "Display Widgets"

    def icon(self):
        #return QtGui.QIcon(_logo_pixmap)
        return MplGraphWidget.get_icon("matplotlib.gif")

    def toolTip(self):
        return "Matplotlib Canvas"

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="MplGraphWidget" name=\"mplgraphwidget\" />\n'
        
    def includeFile(self):
        return "mplgraphwidget"  #Eventually part of matplotlib.backends.backend_qt4agg hopefully


    
_logo_16x16_xpm = [
    "16 16 5 1",
    "B c #000000",
    "r c #ff0000",
    "g c #00ff00",
    "b c #0000ff",
    ". c #ffffff",
    "BBBBBBBBBBBBBBBB",
    "B..............B",
    "B..............B",
    "B..............B",
    "B..............B",
    "B...b..........B",
    "B..b.b.........B",
    "B.b...b........B",
    "Bb.....b.......B",
    "B.......b.....bB",
    "B........b...b.B",
    "B.........b.b..B",
    "B..........b...B",
    "B..............B",
    "B..............B",
    "BBBBBBBBBBBBBBBB"]

_logo_pixmap = QtGui.QPixmap(_logo_16x16_xpm)
