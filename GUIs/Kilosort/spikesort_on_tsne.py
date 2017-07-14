


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import GUIs.Kilosort.correct_pyqtgraph_roi as cpg
import numpy as np
from tsne_for_spikesort import io_with_cpp as io
from os.path import join, realpath
from BrainDataAnalysis import ploting_functions as pf
import sys

base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'
exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
tsne = io.load_tsne_result(exe_folder)
cluster_info = np.load(join(base_folder, 'cluster_info.pkl'))
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

number_of_spikes = 200000  # Or all = spikes_clean_index.size

template_marking = np.load(join(base_folder, 'template_marking.npy'))
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))

clean_templates = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))[:number_of_spikes]
spike_templates_clean = spike_templates[spikes_clean_index]


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        self.pan_mode = False
        self.square_roi_on = 0
        self.freeform_roi_on = 0
        self.freeform_roi_positions = []
        self.roi = None
        self.connect_on_roi_select = None

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.square_roi_on == 0 and self.freeform_roi_on == 0:
                self.autoRange()
            elif self.square_roi_on > 0:
                self.delete_square_roi()
            elif self.freeform_roi_on > 0:
                self.delete_freeform_roi()

        if ev.button() == QtCore.Qt.LeftButton:
            if self.freeform_roi_on > 0:
                position = self.mapToView(ev.pos())
                self.freeform_roi_positions.append(position)
                if self.freeform_roi_on == 1:
                    self.removeItem(self.roi)
                    self.roi = cpg.PolyLineROI(positions=self.freeform_roi_positions, closed=True, movable=False, parent=self)
                    self.roi.sigRegionChangeFinished.connect(self.freeform_roi_handle_move)
                    self.roi.sigClicked.connect(self.freeform_roi_update)
                    self.addItem(self.roi)
                    self.connect_on_roi_select(self.roi.shape(), freeform=True)
                else:
                    self.freeform_roi_update()
                self.freeform_roi_on += 1

        ev.accept()

    def mouseDragEvent(self, ev):
        if self.square_roi_on == 0 and self.freeform_roi_on == 0:
            if ev.button() == QtCore.Qt.RightButton:
                ev.ignore()
            else:
                pg.ViewBox.mouseDragEvent(self, ev)

        elif self.square_roi_on > 0 and self.freeform_roi_on == 0:
            start = self.mapToView(ev.buttonDownPos())
            if self.square_roi_on == 1:
                self.delete_square_roi()
                self.roi = pg.RectROI(start, [0, 0])
                self.addItem(self.roi)
                self.square_roi_on = 2

            if not ev.isStart():
                current = self.mapToView(ev.pos())
                self.roi.setSize(current - start)

            if ev.isFinish():
                self.square_roi_on = 1
                self.connect_on_roi_select(self.roi.parentBounds(), freeform=False)
                ev.ignore()
                return

            ev.accept()

    def set_to_pan_mode(self):
        self.setMouseMode(self.PanMode)
        self.pan_mode = True
        self.delete_square_roi(mode=0)
        self.delete_freeform_roi(mode=0)

    def set_to_rect_mode(self):
        self.setMouseMode(self.RectMode)
        self.pan_mode = False
        self.delete_square_roi(mode=0)
        self.delete_freeform_roi(mode=0)

    def draw_square_roi(self):
        self.square_roi_on = 1
        self.freeform_roi_on = 0
        self.pan_mode = False
        self.delete_freeform_roi(mode=0)

    def draw_freeform_roi(self):
        self.freeform_roi_on = 1
        self.square_roi_on = 0
        self.pan_mode = False
        self.delete_square_roi(mode=0)

    def delete_square_roi(self, mode=1):
        self.square_roi_on = mode
        self.removeItem(self.roi)
        self.roi = None

    def delete_freeform_roi(self, mode=1):
        self.freeform_roi_on = mode
        self.removeItem(self.roi)
        self.roi = None
        self.freeform_roi_positions = []

    def freeform_roi_handle_move(self):
        positions = [name_pos[1] for name_pos in self.roi.getLocalHandlePositions()]
        if len(positions) == len(self.freeform_roi_positions):
            same = True
            for p in range(len(positions)):
                different = 0
                for f in range(len(self.freeform_roi_positions)):
                    if positions[p] != self.freeform_roi_positions[f]:
                        different += 1
                if different == len(self.freeform_roi_positions):
                    same = False
                    break

            if not same:
                self.freeform_roi_positions = positions
                self.roi.setPoints(positions)
                self.connect_on_roi_select(self.roi.shape(), freeform=True)

    def freeform_roi_update(self):
        print(self.freeform_roi_positions)
        self.roi.setPoints(self.freeform_roi_positions, closed=True)
        self.connect_on_roi_select(self.roi.shape(), freeform=True)


app = QtGui.QApplication([])
main_window = QtGui.QMainWindow()

main_toolbar = main_window.addToolBar('Tools')
main_window.setWindowTitle('Spikesort on T-sne')
main_window.resize(1300, 900)
central_window = pg.GraphicsLayoutWidget()
main_window.setCentralWidget(central_window)

custom_viewbox = CustomViewBox()

scatter_plot = central_window.addPlot(0, 0, 3, 2, viewBox=custom_viewbox)
scatter_item = pg.ScatterPlotItem(size=10, pen=None, pxMode=True)
brush = [pg.intColor(spike_templates_clean[i][0], hues=50, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                     sat=255, alpha=255) for i in range(len(tsne))]
tsne_spots = [{'pos': tsne[i, :], 'data': 1, 'brush':brush[i]} for i in range(1000)]
scatter_item.addPoints(tsne_spots)
scatter_plot.addItem(scatter_item)


def on_roi_selection(shape, freeform=False):
    points = np.array([tsne_spots[i]['pos'] for i in range(len(tsne_spots))])
    if not freeform:
        bounds = shape
    else:
        bounds = shape.boundingRect()
    selected = np.argwhere((points[:, 0] > bounds.x()) *
                           (points[:, 0] < bounds.x() + bounds.width()) *
                           (points[:, 1] > bounds.y()) *
                           (points[:, 1] < bounds.y() + bounds.height())).flatten()

    if freeform:
        selected = [i for i in selected if shape.contains(QtCore.QPointF(points[i][0], points[i][1]))]

    # show_selected_points(selected)

    print(len(selected))


def show_selected_points(selected):
    brush_local = list(np.zeros(len(tsne_spots)))
    for i in range(len(tsne_spots)):
        if i in selected:
            brush_local[i] = pg.mkBrush(255, 0, 0, 255)
        else:
            brush_local[i] = brush[i]
    scatter_item.setBrush(brush_local, mask=None)  # VERY SLOW


icons_path = r'E:\Projects\Analysis\Brain\themeaningofbrain\GUIs\Kilosort'
scroll_icon_file = join(sys.path[0], 'scroll_icon2.png')
zoom_icon_file = join(sys.path[0], 'zoom_icon.png')
select_icon_file = join(sys.path[0], 'select_icon.png')
select_freeform_icon_file = join(sys.path[0], 'select_freeform_icon.png')

scroll = QtWidgets.QAction(QtGui.QIcon(scroll_icon_file), "scroll", main_window)
scroll.triggered.connect(custom_viewbox.set_to_pan_mode)
main_toolbar.addAction(scroll)
zoom = QtWidgets.QAction(QtGui.QIcon(zoom_icon_file), "zoom", main_window)
zoom.triggered.connect(custom_viewbox.set_to_rect_mode)
main_toolbar.addAction(zoom)
select = QtWidgets.QAction(QtGui.QIcon(select_icon_file), "select", main_window)
select.triggered.connect(custom_viewbox.draw_square_roi)
main_toolbar.addAction(select)
select_freeform = QtWidgets.QAction(QtGui.QIcon(select_freeform_icon_file), 'select freeform', main_window)
select_freeform.triggered.connect(custom_viewbox.draw_freeform_roi)
custom_viewbox.connect_on_roi_select = on_roi_selection
main_toolbar.addAction(select_freeform)


main_window.show()


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    sys.exit(app.exec_())