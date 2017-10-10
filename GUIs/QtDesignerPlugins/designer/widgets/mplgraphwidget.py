from PyQt4 import QtGui, QtCore

import numpy as np

from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as Canvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import os.path as osp

from matplotlib import rcParams
rcParams['font.size'] = 9

class MplGraphWidget(QtGui.QWidget):

    def __init__(self, parent=None, width=3, height=3, dpi=100):

        super(MplGraphWidget, self).__init__(parent)

        self._dataY = np.array([])
        self._dataX = np.array([])

        self._spCols = 1
        self._spRows = 1
        self.all_sp_axes = []
        self.fig = Figure(figsize=(width,height), dpi=dpi)
        self.all_sp_axes.append(self.fig.add_subplot(self._spCols, self._spRows, 1))
        self.fig.set_frameon(False)
        self.fig.set_tight_layout(True)

        self.canvas = Canvas(self.fig)

        self._navBarOn = True
        self.mpl_toolbar = NavigationToolbar(self.canvas, parent)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        self.canvas.setParent(parent)
        self.canvas.clearMask()
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        if self._navBarOn:
            vbox.addWidget(self.mpl_toolbar)
        self.setLayout(vbox)

    key_pressed = QtCore.pyqtSignal(object, name='keyPressed')

    def on_key_press(self, event):
        self.key_pressed.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    button_pressed = QtCore.pyqtSignal(object, name='buttonPressed')

    def on_button_press(self, event):
        self.button_pressed.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    mouse_move = QtCore.pyqtSignal(object, name='mouseMoved')

    def on_mouse_move(self, event):
        self.mouse_move.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    def get_icon(name):
        """Return Matplotlib icon *name*"""
        return QtGui.QIcon(osp.join(rcParams['datapath'], 'images', name))

    def generateNewAxes(self):
        for ax in self.all_sp_axes:
            self.fig.delaxes(ax)
        self.all_sp_axes = []
        numOfAxes = (self._spRows*self._spCols)+1
        for i in np.arange(1,numOfAxes):
            self.all_sp_axes.append(self.fig.add_subplot(self._spRows, self._spCols, i))
        self.canvas.setGeometry(100, 100, 300, 300)  #Used to update the new number of axes
        self.canvas.updateGeometry()  #This will bring the size of the canvas back to the original (defined by the vbox)

    spRowsChanged = QtCore.pyqtSignal(int)

    def getspRows(self):
        return self._spRows

    @QtCore.pyqtSlot(int)
    def setspRows(self, spRows):
        self._spRows = spRows
        self.generateNewAxes()
        self.spRowsChanged.emit(spRows)

    def resetspRows(self):
        self.setspRows(1)

    spRows = QtCore.pyqtProperty(int, getspRows, setspRows, resetspRows)

    spColsChanged = QtCore.pyqtSignal(int)

    def getspCols(self):
        return self._spCols

    @QtCore.pyqtSlot(int)
    def setspCols(self, spCols):
        self._spCols = spCols
        self.generateNewAxes()
        self.spRowsChanged.emit(spCols)

    def resetspCols(self):
        self.setspCols(1)

    spCols = QtCore.pyqtProperty(int, getspCols, setspCols, resetspCols)

    dataChanged = QtCore.pyqtSignal(bool)

    def get_Y_data(self):
        return self._dataY

    @QtCore.pyqtSlot(int)
    def set_Y_data(self, y_data):
        self._dataY = y_data
        self.dataChanged.emit(True)


    def plot(self, on_axes=0):
        if np.size(self._dataX) == 0:
            self.all_sp_axes[on_axes].plot(self._dataY)
        else:
            self.all_sp_axes[on_axes].plot(self._dataX, self._dataY)

    def getNavBarOn(self):
        return self._navBarOn

    def setNavBarOn(self, navBarOn):
        self._navBarOn = navBarOn
        if not navBarOn:
            self.mpl_toolbar.hide()
        else:
            self.mpl_toolbar.show()

    def resetNavBarOn(self):
        self._navBarOn = True

    navBarOn = QtCore.pyqtProperty(bool, getNavBarOn, setNavBarOn, resetNavBarOn)


    # def getautoscale(self):
    #     return self._autoscale
    #
    # def setautoscale(self, autoscale):
    #     self._autoscale = autoscale
    #     for axis in self.all_sp_axes:
    #         axis.set_autoscale(autoscale)
    #
    # def resetautoscale(self):
    #     self._autoscale = False
    #
    # autoscale = QtCore.pyqtProperty(bool, getautoscale, setautoscale, resetautoscale)

    @QtCore.pyqtSlot(bool)
    def set_autoscale(self, autoscale):
        for axis in self.all_sp_axes:
            axis.set_autoscale(autoscale)








if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    widget = MplGraphWidget()
    widget.show()
    sys.exit(app.exec_())