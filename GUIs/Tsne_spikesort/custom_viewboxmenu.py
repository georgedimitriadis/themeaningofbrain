from pyqtgraph.Qt import QtCore, QtGui, QT_LIB
from pyqtgraph.python2_3 import asUnicode
from pyqtgraph.WidgetGroup import WidgetGroup

if QT_LIB == 'PyQt4':
    from pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt import Ui_Form as AxisCtrlTemplate
elif QT_LIB == 'PySide':
    from pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyside import Ui_Form as AxisCtrlTemplate
elif QT_LIB == 'PyQt5':
    from pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5 import Ui_Form as AxisCtrlTemplate

import weakref


class ViewBoxMenu(QtGui.QMenu):
    def __init__(self, view):
        QtGui.QMenu.__init__(self)

        self.view = view

        self.setTitle("ViewBox options")
        self.add_roi = QtGui.QAction('Add Extra ROI', self)
        self.addAction(self.add_roi)

        self.add_roi.triggered.connect(self.on_add_roi)

    def on_add_roi(self):
        self.view.number_of_rois += 1

