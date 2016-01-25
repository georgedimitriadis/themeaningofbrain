from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSizePolicy


import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import os.path as osp
from matplotlib import rcParams
rcParams['font.size'] = 9

class MplGraphQt5Widget(QWidget):
    def __init__(self, parent=None):
        super(MplGraphQt5Widget, self).__init__(parent)

        self.width = 3
        self.height = 3
        self.dpi = 100

        self._dataY = np.array([])
        self._dataX = np.array([])

        self._spCols = 1
        self._spRows = 1
        self.all_sp_axes = []
        self.fig = Figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.all_sp_axes.append(self.fig.add_subplot(self._spCols, self._spRows, 1))
        self.fig.set_frameon(False)
        self.fig.set_tight_layout(True)

        self.canvas = Canvas(self.fig)

        self._navBarOn = False
        self.mpl_toolbar = NavigationToolbar(self.canvas, parent)
        self.mpl_toolbar.dynamic_update()

        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        self.canvas.setParent(parent)
        self.canvas.clearMask()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        if not self._navBarOn:
            self.mpl_toolbar.hide()
        self.setLayout(vbox)



    def get_icon(name):
        """Return Matplotlib icon *name*"""
        return QIcon(osp.join(rcParams['datapath'], 'images', name))


    key_pressed = pyqtSignal(object, name='keyPressed')

    def on_key_press(self, event):
        self.key_pressed.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    button_pressed = pyqtSignal(object, name='buttonPressed')

    def on_button_press(self, event):
        self.button_pressed.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    mouse_move = pyqtSignal(object, name='mouseMoved')

    def on_mouse_move(self, event):
        self.mouse_move.emit(event)
        key_press_handler(event, self.canvas, self.mpl_toolbar)


    def generateNewAxes(self):
        for ax in self.all_sp_axes:
            self.fig.delaxes(ax)
        self.all_sp_axes = []
        numOfAxes = (self._spRows*self._spCols)+1
        for i in np.arange(1,numOfAxes):
            self.all_sp_axes.append(self.fig.add_subplot(self._spRows, self._spCols, i))
        self.canvas.setGeometry(100, 100, 300, 300)  #Used to update the new number of axes
        self.canvas.updateGeometry()  #This will bring the size of the canvas back to the original (defined by the vbox)

    spRowsChanged = pyqtSignal(int)

    def getspRows(self):
        return self._spRows

    @pyqtSlot(int)
    def setspRows(self, spRows):
        self._spRows = spRows
        self.generateNewAxes()
        self.spRowsChanged.emit(spRows)

    def resetspRows(self):
        self.setspRows(1)

    spRows = pyqtProperty(int, getspRows, setspRows, resetspRows)

    spColsChanged = pyqtSignal(int)

    def getspCols(self):
        return self._spCols

    @pyqtSlot(int)
    def setspCols(self, spCols):
        self._spCols = spCols
        self.generateNewAxes()
        self.spRowsChanged.emit(spCols)

    def resetspCols(self):
        self.setspCols(1)

    spCols = pyqtProperty(int, getspCols, setspCols, resetspCols)

    dataChanged = pyqtSignal(bool)

    def get_Y_data(self):
        return self._dataY

    @pyqtSlot(int)
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

    navBarOn = pyqtProperty(bool, getNavBarOn, setNavBarOn, resetNavBarOn)

    @pyqtSlot(bool)
    def set_autoscale(self, autoscale):
        for axis in self.all_sp_axes:
            axis.set_autoscale(autoscale)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    widget = MplGraphQt5Widget()
    widget.show()
    sys.exit(app.exec_())
    #app.exec()
