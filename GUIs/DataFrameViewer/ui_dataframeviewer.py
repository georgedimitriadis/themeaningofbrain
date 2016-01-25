# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DataFrameViewer.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PandasDataFrameViewer(object):
    def setupUi(self, PandasDataFrameViewer):
        PandasDataFrameViewer.setObjectName("PandasDataFrameViewer")
        PandasDataFrameViewer.resize(766, 640)
        self.centralwidget = QtWidgets.QWidget(PandasDataFrameViewer)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.table_dataframe_viewer = QtWidgets.QTableWidget(self.centralwidget)
        self.table_dataframe_viewer.setAlternatingRowColors(True)
        self.table_dataframe_viewer.setRowCount(0)
        self.table_dataframe_viewer.setColumnCount(0)
        self.table_dataframe_viewer.setObjectName("table_dataframe_viewer")
        self.table_dataframe_viewer.horizontalHeader().setCascadingSectionResizes(True)
        self.table_dataframe_viewer.horizontalHeader().setSortIndicatorShown(True)
        self.table_dataframe_viewer.horizontalHeader().setStretchLastSection(True)
        self.table_dataframe_viewer.verticalHeader().setCascadingSectionResizes(True)
        self.table_dataframe_viewer.verticalHeader().setSortIndicatorShown(True)
        self.table_dataframe_viewer.verticalHeader().setStretchLastSection(True)
        self.verticalLayout.addWidget(self.table_dataframe_viewer)
        PandasDataFrameViewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PandasDataFrameViewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 766, 21))
        self.menubar.setObjectName("menubar")
        PandasDataFrameViewer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PandasDataFrameViewer)
        self.statusbar.setObjectName("statusbar")
        PandasDataFrameViewer.setStatusBar(self.statusbar)

        self.retranslateUi(PandasDataFrameViewer)
        QtCore.QMetaObject.connectSlotsByName(PandasDataFrameViewer)

    def retranslateUi(self, PandasDataFrameViewer):
        _translate = QtCore.QCoreApplication.translate
        PandasDataFrameViewer.setWindowTitle(_translate("PandasDataFrameViewer", "MainWindow"))
        self.table_dataframe_viewer.setSortingEnabled(True)

