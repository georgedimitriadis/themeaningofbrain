

# Test Stuff -----------------
# ----------------------------
import numpy as np
from os.path import join
import pandas as pd
from BrainDataAnalysis import neuroseeker_specific_functions as ns

base_folder = r'F:\Neuroseeker\\' + \
              r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
              r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Desktop
# base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Laptop

files_folder = join(base_folder, 'Tsne_Results', '2017_08_20_dis700K_579templates') # Desktop
# files_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release' # Laptop

spike_info = pd.read_pickle(join(files_folder, 'spike_info.df'))
channel_map = np.squeeze(np.load(join(base_folder, 'channel_map.npy')))
sampling_frequency = 20000
number_of_channels_in_binary_file = 1440
duration_of_time_plot = 0.004
binary_data_filename = join(base_folder, 'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin')
raw_data = ns.load_binary_amplifier_data(binary_data_filename)
prb_file = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker\prb.txt'
# ----------------------------
# ----------------------------


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
from os.path import join
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from pynput import keyboard
try:
    from . import custom_viewbox
except:  # Exception
    from GUIs.Tsne_spikesort import custom_viewbox
try:
    from . import matplotlib_widget
except:  # Exception
    from GUIs.Tsne_spikesort import matplotlib_widget
try:
    from . import helper_functions as hf
except:  # Exception
    from GUIs.Tsne_spikesort import helper_functions as hf
try:
    from . import spike_heatmap as sh
except:  # Exception
    from GUIs.Tsne_spikesort import spike_heatmap as sh

global currently_selected
currently_selected = np.empty(0)

global previously_selected
previously_selected = np.empty(0)

global currently_selected_templates
currently_selected_templates = np.empty(0)

global avg_electrode_curves
avg_electrode_curves = []

global data_of_selected_spikes
data_of_selected_spikes = np.empty(0)


def on_roi_selection(shape, freeform=False):
    global currently_selected
    global currently_selected_templates
    global previously_selected

    points = np.array([spike_info['tsne_x'].iloc[:number_of_spikes].tolist(),
                      spike_info['tsne_y'].iloc[:number_of_spikes].tolist()]).T
    #points = np.array([tsne_spots[i]['pos'] for i in range(len(tsne_spots))])

    previously_selected = currently_selected

    if not freeform:
        bounds = shape
    else:
        bounds = shape.boundingRect()

    selected = np.argwhere((points[:, 0] > bounds.x()) *
                           (points[:, 0] < bounds.x() + bounds.width()) *
                           (points[:, 1] > bounds.y()) *
                           (points[:, 1] < bounds.y() + bounds.height())).flatten()
    if freeform:
        currently_selected = [i for i in selected if shape.contains(QtCore.QPointF(points[i][0], points[i][1]))]
    else:
        currently_selected = selected

    currently_selected_templates = hf.find_templates_of_spikes(spike_info, currently_selected)
    print('Number of spikes selected: ' + str(len(selected)))

    #show_selected_points()


def on_roi_deletion():
    global currently_selected
    global currently_selected_templates

    currently_selected = np.empty(0)
    currently_selected_templates = np.empty(0)


def show_selected_points():
    points = scatter_item.points()
    for i in range(number_of_spikes):
        if i in currently_selected:
            points[i].setBrush(pg.mkBrush(255, 255, 255, 255))
        elif i in previously_selected:
            points[i].setBrush(brush[i])
    print('hello')
    #scatter_item.setSymbolBrush(brush_local)
    #scatter_item.setBrush(brush_local, mask=None)  # VERY SLOW


def show_selected_spikes():
    #sizes = np.ones(number_of_spikes) * 7
    #sizes[currently_selected] = 14
    #scatter_item.setSize(sizes)
    #for i in currently_selected:
    #    tsne_spots[i]['brush'] = pg.mkBrush(255, 255, 255, 255)
    #scatter_item.updateSpots()
    pass

def update_average_spikes_plot():
    global currently_selected
    global data_of_selected_spikes

    half_duration = duration_of_time_plot / 2
    time_points = int(half_duration * sampling_frequency)
    time_axis = np.arange(-half_duration, half_duration, 1 / sampling_frequency)

    data_of_selected_spikes = np.zeros((electrodes, 2 * time_points))
    times = spike_info['times'].iloc[currently_selected]

    progdialog = QtWidgets.QProgressDialog()
    progdialog.setMinimum(0)
    progdialog.setMaximum(len(times))
    progdialog.setWindowTitle('Loading spikes')
    progdialog.setGeometry(500, 500, 400, 40)
    progdialog.setWindowModality(QtCore.Qt.WindowModal)

    progdialog.show()

    prog = 0
    for t in times:
        add_data = raw_data[channel_map, int(t-time_points):int(t+time_points)]
        data_of_selected_spikes = data_of_selected_spikes + add_data
        prog += 1
        progdialog.setValue(prog)
        progdialog.setLabelText('Loading spike: ' + str(prog) + ' of ' + str(len(times)))
        QtWidgets.QApplication.processEvents()

    data_of_selected_spikes = data_of_selected_spikes / len(times)

    for i in np.arange(electrodes):
        avg_electrode_curves[i].setData(time_axis, data_of_selected_spikes[i, :])
        avg_electrode_curves[i].setPen(pg.mkPen(color=(i, number_of_channels_in_binary_file * 1.3),
                                                width=0.5))
    progdialog.setValue(len(times))
    progdialog.close()


def update_heatmap_plot():
    global data_of_selected_spikes

    connected = np.squeeze(np.load(join(base_folder, 'channel_map.npy')))
    connected_binary = np.in1d(np.arange(number_of_channels_in_binary_file), connected)
    bad_channels = np.squeeze(np.argwhere(connected_binary==False).astype(np.int))

    sh.create_heatmap_on_matplotlib_widget(heatmap_plot, data_of_selected_spikes, prb_file, window_size=60,
                                           bad_channels=bad_channels, num_of_shanks=5, rotate_90=True, flip_ud=False,
                                           flip_lr=False)
    heatmap_plot.draw()


def update_autocorelogram():
    global currently_selected
    spike_times = spike_info['times'].iloc[currently_selected].as_matrix().astype(np.int64)
    diffs, norm = hf.crosscorrelate_spike_trains(spike_times, spike_times, lag=3000)
    hist, edges = np.histogram(diffs, bins=100)
    autocorelogram_curve.setData(x=edges, y=hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))


# On press button functions ---------
def on_press_button_update():
    update_average_spikes_plot()
    update_autocorelogram()
    update_heatmap_plot()


def on_tsne_color_scheme_combo_box_change(index):
    progdialog = QtWidgets.QProgressDialog()
    progdialog.setGeometry(500, 500, 400, 40)
    progdialog.setWindowTitle('Changing spike color scheme')
    progdialog.setMinimum(0)
    progdialog.setMaximum(2)
    progdialog.setWindowModality(QtCore.Qt.WindowModal)

    progdialog.setLabelText('Generating colors and symbols ...                  ')
    progdialog.show()
    progdialog.setValue(0)
    QtWidgets.QApplication.processEvents()

    spikes_to_change = range(number_of_spikes)

    if index == 0:

        color_scheme = spike_info['template_after_cleaning']
        brush = [pg.intColor(color_scheme.iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                             maxHue=360, minHue=0, sat=255, alpha=255)
                 for i in spikes_to_change]
        symbol = []
        for i in spikes_to_change:
            symbol.append(hf.symbol_from_type(spike_info['type_after_cleaning'].iloc[i]))
        '''
        symbol = []
        for i in range(number_of_spikes):
            symbol.append(hf.symbol_from_type(spike_info['type_after_cleaning'].iloc[i]))
        color = [
            pg.intColor(spike_info['type_after_cleaning'].iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                        maxHue=360, minHue=0,
                        sat=255, alpha=255) for i in range(number_of_spikes)]
        brush = [pg.mkBrush(color[i]) for i in range(number_of_spikes)]
        '''
    elif index == 1:

        color_scheme = spike_info['template_after_sorting']
        brush = [pg.intColor(color_scheme.iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                             maxHue=360, minHue=0, sat=255, alpha=255)
                 for i in spikes_to_change]
        symbol = []
        for i in spikes_to_change:
            symbol.append(hf.symbol_from_type(spike_info['type_after_sorting'].iloc[i]))
        '''
        symbol = []
        for i in range(number_of_spikes):
            symbol.append(hf.symbol_from_type(spike_info['type_after_sorting'].iloc[i]))
        color = [
            pg.intColor(spike_info['template_after_sorting'].iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                        maxHue=360, minHue=0,
                        sat=255, alpha=255) for i in range(number_of_spikes)]
        brush = [pg.mkBrush(color[i]) for i in range(number_of_spikes)]
        '''
    progdialog.setValue(1)
    progdialog.setLabelText('Applying colors and symbols ...                   ')
    QtWidgets.QApplication.processEvents()

    #scatter_item.setSymbolBrush(brush)
    scatter_item.setBrush(brush)
    scatter_item.setSymbol(symbol)

    progdialog.setValue(2)
    progdialog.close()


def on_templates_table_selection(row, column):
    template_selected = int(templates_table.item(row, 0).text())
    print(template_selected)

# ----------------------------------


# Main Window ----------------------
app = QtGui.QApplication([])
main_window = QtGui.QMainWindow()

main_toolbar = main_window.addToolBar('Tools')
main_window.setWindowTitle('Spikesort on T-sne')
main_window.resize(1300, 900)
central_window = pg.GraphicsLayoutWidget()
main_window.setCentralWidget(central_window)
grid_layout = QtGui.QGridLayout()
central_window.setLayout(grid_layout)
# ----------------------------------

# T-sne Scatter Plot ---------------
custom_viewbox = custom_viewbox.CustomViewBox()
scatter_plot = pg.PlotWidget(title='T-sne plot', viewBox=custom_viewbox)
grid_layout.addWidget(scatter_plot, 0, 0, 2, 4)

scatter_plot.plotItem.ctrlMenu = None
scatter_plot.scene().contextMenu = None

# scatter_item = custom_plotdataitem.CustomPlotDataItem(size=7, pen=None, pxMode=True)
scatter_item = pg.ScatterPlotItem(size=7, pen=None, pxMode=True)

number_of_spikes = 10000#len(spike_info)

color = [pg.intColor(spike_info['template_after_sorting'].iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                     maxHue=360, minHue=0,
                     sat=255, alpha=255) for i in range(number_of_spikes)]
brush = [pg.mkBrush(color[i]) for i in range(number_of_spikes)]

tsne_spots = [{'pos': [spike_info['tsne_x'].iloc[i], spike_info['tsne_y'].iloc[i]],
               'data': 1, 'brush':brush[i]} for i in range(number_of_spikes)]
scatter_item.addPoints(tsne_spots)
# scatter_item.setData(x=spike_info['tsne_x'].iloc[:number_of_spikes].tolist(),
#                     y=spike_info['tsne_y'].iloc[:number_of_spikes].tolist())

symbol = []
for i in range(number_of_spikes):
    symbol.append(hf.symbol_from_type(spike_info['type_after_sorting'].iloc[i]))
scatter_item.setSymbol(symbol)

# scatter_item.setSymbolBrush(brush)

scatter_plot.addItem(scatter_item)
# ----------------------------------

# Average spike time course plot ---
spikes_time_plot = pg.PlotWidget(title='Average time plot of selected spikes')
grid_layout.addWidget(spikes_time_plot, 2, 0, 3, 2)
electrodes = len(channel_map)
for i in range(electrodes):
    electrode_curve = pg.PlotCurveItem()
    spikes_time_plot.addItem(electrode_curve)
    avg_electrode_curves.append(electrode_curve)
# ----------------------------------

# Autocorelogram plot --------
autocorelogram_plot = pg.PlotWidget(title='Autocorrelogram')
grid_layout.addWidget(autocorelogram_plot, 2, 2, 3, 2)
autocorelogram_curve = pg.PlotCurveItem()
autocorelogram_plot.addItem(autocorelogram_curve)
# ----------------------------

# Heatmap plot ---------------
heatmap_plot = matplotlib_widget.MatplotlibWidget(toolbar_on=False)
grid_layout.addWidget(heatmap_plot, 0, 4, 6, 2)
# ----------------------------

# Templates Table widget -----
templates_table = pg.TableWidget(editable=False, sortable=True)
templates_table.setData(hf.get_templates_with_number_of_spikes_from_spike_info(spike_info))
templates_table.setHorizontalHeaderLabels(['Template number', 'Number of spikes in template'])
header = templates_table.horizontalHeader()
header.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
header.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)
templates_table.cellClicked.connect(on_templates_table_selection)
grid_layout.addWidget(templates_table, 0, 6, 3, 1)
# ----------------------------

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
custom_viewbox.connect_on_roi_delete = on_roi_deletion
main_toolbar.addAction(select_freeform)
# ----------------------------------

# Buttons --------------------
button_update = QtGui.QPushButton('Update')
button_update.clicked.connect(on_press_button_update)
grid_layout.addWidget(button_update, 7, 0, 1, 1)

tsne_color_scheme_combo_box = QtGui.QComboBox()
tsne_color_scheme_combo_box.addItem('After sorting')
tsne_color_scheme_combo_box.addItem('Before sorting')
tsne_color_scheme_combo_box.activated.connect(on_tsne_color_scheme_combo_box_change)
grid_layout.addWidget(tsne_color_scheme_combo_box, 7, 1, 1, 1)
# ----------------------------------

main_window.show()


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    sys.exit(app.exec_())

