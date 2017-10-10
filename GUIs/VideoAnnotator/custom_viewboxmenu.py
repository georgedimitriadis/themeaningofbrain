import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QT_LIB



class ViewBoxMenu(QtGui.QMenu):
    def __init__(self, view):
        QtGui.QMenu.__init__(self)

        self.view = view

        self.setTitle("ViewBox options")
        self.add_red_roi = QtGui.QAction('Add Red ROI', self)
        self.addAction(self.add_red_roi)
        self.add_red_roi.triggered.connect(self.on_add_red_roi)
        self.add_green_roi = QtGui.QAction('Add Green ROI', self)
        self.addAction(self.add_green_roi)
        self.add_green_roi.triggered.connect(self.on_add_green_roi)


    def on_add_red_roi(self):
        self.view.number_of_rois += 1
        self.view.current_roi_color = 'r'

    def on_add_green_roi(self):
        self.view.number_of_rois += 1
        self.view.current_roi_color = 'g'
