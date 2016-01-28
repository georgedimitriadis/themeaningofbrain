# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tfr_viewer.ui'
#
# Created: Wed Apr  8 17:43:27 2015
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_TFR_Viewer(object):
    def setupUi(self, TFR_Viewer):
        TFR_Viewer.setObjectName(_fromUtf8("TFR_Viewer"))
        TFR_Viewer.resize(1054, 671)
        self.centralwidget = QtGui.QWidget(TFR_Viewer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.l_channel = QtGui.QLabel(self.centralwidget)
        self.l_channel.setGeometry(QtCore.QRect(10, 599, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.l_channel.setFont(font)
        self.l_channel.setObjectName(_fromUtf8("l_channel"))
        self.sb_channel = QtGui.QSpinBox(self.centralwidget)
        self.sb_channel.setGeometry(QtCore.QRect(100, 600, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sb_channel.setFont(font)
        self.sb_channel.setMaximum(256)
        self.sb_channel.setObjectName(_fromUtf8("sb_channel"))
        self.mplg_tfr_viewer = MplGraphWidget(self.centralwidget)
        self.mplg_tfr_viewer.setGeometry(QtCore.QRect(-10, -10, 1071, 611))
        self.mplg_tfr_viewer.setObjectName(_fromUtf8("mplg_tfr_viewer"))
        TFR_Viewer.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(TFR_Viewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1054, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        TFR_Viewer.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(TFR_Viewer)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        TFR_Viewer.setStatusBar(self.statusbar)

        self.retranslateUi(TFR_Viewer)
        QtCore.QMetaObject.connectSlotsByName(TFR_Viewer)

    def retranslateUi(self, TFR_Viewer):
        TFR_Viewer.setWindowTitle(_translate("TFR_Viewer", "TFR Viewer", None))
        self.l_channel.setText(_translate("TFR_Viewer", "Channel:", None))

from mplgraphwidget import MplGraphWidget
