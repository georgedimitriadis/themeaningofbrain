# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tfr_viewer_qt5.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TFR_Viewer(object):
    def setupUi(self, TFR_Viewer):
        TFR_Viewer.setObjectName("TFR_Viewer")
        TFR_Viewer.resize(1117, 692)
        self.centralwidget = QtWidgets.QWidget(TFR_Viewer)
        self.centralwidget.setObjectName("centralwidget")
        self.l_channel = QtWidgets.QLabel(self.centralwidget)
        self.l_channel.setGeometry(QtCore.QRect(10, 599, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.l_channel.setFont(font)
        self.l_channel.setObjectName("l_channel")
        self.sb_channel = QtWidgets.QSpinBox(self.centralwidget)
        self.sb_channel.setGeometry(QtCore.QRect(100, 600, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sb_channel.setFont(font)
        self.sb_channel.setMaximum(256)
        self.sb_channel.setObjectName("sb_channel")
        self.mplg_tfr_viewer = MplGraphQt5Widget(self.centralwidget)
        self.mplg_tfr_viewer.setGeometry(QtCore.QRect(-10, -10, 1071, 611))
        self.mplg_tfr_viewer.setObjectName("mplg_tfr_viewer")
        TFR_Viewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TFR_Viewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1117, 31))
        self.menubar.setObjectName("menubar")
        TFR_Viewer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TFR_Viewer)
        self.statusbar.setObjectName("statusbar")
        TFR_Viewer.setStatusBar(self.statusbar)

        self.retranslateUi(TFR_Viewer)
        QtCore.QMetaObject.connectSlotsByName(TFR_Viewer)

    def retranslateUi(self, TFR_Viewer):
        _translate = QtCore.QCoreApplication.translate
        TFR_Viewer.setWindowTitle(_translate("TFR_Viewer", "TFR Viewer"))
        self.l_channel.setText(_translate("TFR_Viewer", "Channel:"))

from mplgraphqt5widget import MplGraphQt5Widget
