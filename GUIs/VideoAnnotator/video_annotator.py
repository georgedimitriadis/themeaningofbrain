

import cv2
from os.path import join
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import sys
try:
    from . import custom_viewbox
except:  # Exception
    from GUIs.VideoAnnotator import custom_viewbox

data_folder = r'D:\Data\Danby\L1-H2013-03_2014-09-09_TS02'
video_file = r'L1-H2013-03_2014-09-09_TS02_raw.avi'


cap = cv2.VideoCapture(join(data_folder, video_file))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1000 / frame_rate

global current_frame
current_frame = 0


def get_image_at_frame(frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    err, frame = cap.read()
    frame = np.flipud(frame)
    return frame

def on_go_to_frame_text_edited(text):
    global current_frame

    if len(text) > 0:
        frame_to_go = int(text)
        if frame_to_go > -1 and frame_to_go <= length:
            image_item.setImage(get_image_at_frame(frame_to_go))
            frame_number_text.setText('Frame = ' + str(frame_to_go))
            current_frame = frame_to_go


def on_press_button_random_forwards():
    global current_frame

    frames_remaining = length - current_frame
    frame_to_go = int(frames_remaining * np.random.random() + current_frame)
    image_item.setImage(get_image_at_frame(frame_to_go))
    frame_number_text.setText('Frame = ' + str(frame_to_go))
    current_frame = frame_to_go


def on_press_button_random_backwards():
    global current_frame

    frame_to_go = int(current_frame * np.random.random())
    image_item.setImage(get_image_at_frame(frame_to_go))
    frame_number_text.setText('Frame = ' + str(frame_to_go))
    current_frame = frame_to_go


def on_roi_selection(roi, freeform=False):
    pass

def on_roi_deletion():
    pass


pg.setConfigOptions(imageAxisOrder='row-major')
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


# Image Plot -----------------------
custom_viewbox = custom_viewbox.CustomViewBox()
image_plot = pg.PlotWidget(title='Frame', viewBox=custom_viewbox)
grid_layout.addWidget(image_plot, 0, 0, 4, 4)

image_plot.plotItem.ctrlMenu = None
image_plot.scene().contextMenu = None

image_item = pg.ImageItem()
image_item.setImage(get_image_at_frame(0))
image_plot.addItem(image_item)
frame_number_text = pg.TextItem('Frame = ' + str(current_frame))
image_plot.addItem(frame_number_text)
frame_number_text.setPos(0, 0)


# Inputs --------------------------
button_forwards = QtGui.QPushButton('Move forwards randomly')
button_forwards.clicked.connect(on_press_button_random_forwards)
grid_layout.addWidget(button_forwards, 5, 2, 1, 1)

button_backwards = QtGui.QPushButton('Move backwards randomly')
button_backwards.clicked.connect(on_press_button_random_backwards)
grid_layout.addWidget(button_backwards, 5, 1, 1, 1)


go_to_frame_edit = QtWidgets.QLineEdit()
go_to_frame_edit.textChanged.connect(on_go_to_frame_text_edited)
grid_layout.addWidget(go_to_frame_edit, 5, 0, 1, 1)


# Tool bar -------------------------
icons_path = r'E:\Projects\Analysis\Brain\themeaningofbrain\GUIs\VideoAnnotator'
select_icon_file = join(sys.path[0], 'Icons',  'select_icon.png')
select_freeform_icon_file = join(sys.path[0], 'Icons',  'select_freeform_icon.png')

select = QtWidgets.QAction(QtGui.QIcon(select_icon_file), "select", main_window)
select.triggered.connect(custom_viewbox.draw_square_roi)
main_toolbar.addAction(select)
select_freeform = QtWidgets.QAction(QtGui.QIcon(select_freeform_icon_file), 'select freeform', main_window)
select_freeform.triggered.connect(custom_viewbox.draw_freeform_roi)
custom_viewbox.connect_on_roi_select = on_roi_selection
custom_viewbox.connect_on_roi_delete = on_roi_deletion
main_toolbar.addAction(select_freeform)
# ----------------------------------


main_window.show()


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    sys.exit(app.exec_())