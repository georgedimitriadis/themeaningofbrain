__author__ = 'George Dimitriadis'



from PyQt5 import QtWidgets
from GUIs.DataFrameViewer.ui_dataframeviewer import Ui_PandasDataFrameViewer
import sys


def view_dataframe(df):
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_PandasDataFrameViewer()
    ui.setupUi(window)

    datatable = ui.table_dataframe_viewer
    datatable.setRowCount(len(df.index))
    datatable.setVerticalHeaderLabels([str(x) for x in df.index])

    name = df.__module__
    if "series" in name:
        datatable.setColumnCount(1)
        datatable.setHorizontalHeaderLabels("Series")
        for i in range(len(df.index)):
            datatable.setItem(i, 0, QtWidgets.QTableWidgetItem(str(df.iget_value(i))))
    elif "frame" in name:
        datatable.setColumnCount(len(df.columns))
        datatable.setHorizontalHeaderLabels([str(x) for x in df.columns])
        datatable.setVerticalHeaderLabels([str(x) for x in df.index])
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                datatable.setItem(i, j, QtWidgets.QTableWidgetItem(str(df.iget_value(i, j))))

    window.show()
    app.exec_()