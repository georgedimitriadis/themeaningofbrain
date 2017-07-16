
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
from os.path import join
import sys
try:
    from . import custom_viewbox
except SystemError:  # Exception
    from GUIs.Tsne_spikesort import custom_viewbox


# Test Stuff -----------------
#base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_03_28_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Desktop
base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Laptop

#exe_folder = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release' # Desktop
exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release' # Laptop

from tsne_for_spikesort import io_with_cpp as io
tsne = io.load_tsne_result(exe_folder)
cluster_info = np.load(join(base_folder, 'cluster_info.pkl'))
from BrainDataAnalysis import ploting_functions as pf
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

number_of_spikes = 200000  # Or all = spikes_clean_index.size

template_marking = np.load(join(base_folder, 'template_marking.npy'))
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))

clean_templates = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))[:number_of_spikes]
spike_templates_clean = spike_templates[spikes_clean_index]
# ----------------------------




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
    #show_selected_points(selected)
    print(len(selected))


def show_selected_points(selected):
    brush_local = list(np.zeros(len(tsne_spots)))
    for i in range(len(tsne_spots)):
        if i in selected:
            brush_local[i] = pg.mkBrush(255, 0, 0, 255)
        else:
            brush_local[i] = brush[i]
    scatter_item.setBrush(brush_local, mask=None)  # VERY SLOW


# Main Window ----------------------
app = QtGui.QApplication([])
main_window = QtGui.QMainWindow()

main_toolbar = main_window.addToolBar('Tools')
main_window.setWindowTitle('Spikesort on T-sne')
main_window.resize(1300, 900)
central_window = pg.GraphicsLayoutWidget()
main_window.setCentralWidget(central_window)
# ----------------------------------

# T-sne Scatter Plot ---------------
custom_viewbox = custom_viewbox.CustomViewBox()
scatter_plot = central_window.addPlot(0, 0, 3, 2, viewBox=custom_viewbox)
scatter_item = pg.ScatterPlotItem(size=10, pen=None, pxMode=True)
brush = [pg.intColor(spike_templates_clean[i][0], hues=50, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                     sat=255, alpha=255) for i in range(len(tsne))]
tsne_spots = [{'pos': tsne[i, :], 'data': 1, 'brush':brush[i]} for i in range(1000)]
scatter_item.addPoints(tsne_spots)
scatter_plot.addItem(scatter_item)
# ----------------------------------

# Average spike time course plot ---


# ----------------------------------


# Tool bar -------------------------
icons_path = r'E:\Projects\Analysis\Brain\themeaningofbrain\GUIs\Kilosort'
scroll_icon_file = join(sys.path[0], 'Icons',  'scroll_icon.png')
zoom_icon_file = join(sys.path[0], 'Icons', 'zoom_icon.png')
select_icon_file = join(sys.path[0], 'Icons',  'select_icon.png')
select_freeform_icon_file = join(sys.path[0], 'Icons',  'select_freeform_icon.png')

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
# ----------------------------------

main_window.show()


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    sys.exit(app.exec_())