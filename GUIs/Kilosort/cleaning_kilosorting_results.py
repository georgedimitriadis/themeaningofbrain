import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from BrainDataAnalysis import ploting_functions as pf
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# ---- Some basic input--
base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'
binary_data_filename = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'
number_of_channels_in_binary_file = 1440
# -----------------------


def cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename,
                            overwrite_avg_spike_template_file=False, overwrite_template_marking_file=False, freq=20000,
                            time_points=100, figure_id=0, timeToPlot=None):

    channel_map = np.load(os.path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(os.path.join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(os.path.join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    data_raw = np.memmap(os.path.join(base_folder, binary_data_filename),
                         dtype=np.int16, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_kilosorted = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

    spike_times = np.load(os.path.join(base_folder, 'spike_times.npy')).astype(np.int)

    templates = np.load(os.path.join(base_folder, 'templates.npy'))

    num_of_channels = active_channel_map.size

    global current_template_index
    current_template_index = 0
    global visibility_threshold
    visibility_threshold = 2

    if os.path.exists(os.path.join(base_folder, 'template_marking.npy')) and not overwrite_template_marking_file:
        template_marking = np.load(os.path.join(base_folder, 'template_marking.npy'))
    else:
        template_marking = np.zeros(number_of_templates)
        np.save(os.path.join(base_folder, 'template_marking.npy'), template_marking)

    if os.path.exists(os.path.join(base_folder, 'avg_spike_template.npy')) and not overwrite_avg_spike_template_file:
        data = np.load(os.path.join(base_folder, 'avg_spike_template.npy'))
        time_points = int(data.shape[2] / 2)
    else:
        data = np.zeros((number_of_templates, num_of_channels, time_points * 2))

    # -------Callback Functions---------------
    def on_next():
        global current_template_index
        if current_template_index < data.shape[0] - 1:
            current_template_index += 1
            template = np.argwhere(spike_templates == current_template_index)
            num_of_spikes_in_template = template.shape[0]
            while num_of_spikes_in_template == 0 and current_template_index < data.shape[0] - 1:
                current_template_index += 1
                template = np.argwhere(spike_templates == current_template_index)
                num_of_spikes_in_template = template.shape[0]
            lines_multi_data = ax_multidim_data.get_lines()
            y = generate_data(current_template_index)
            visible_channels = get_visible_channels()
            for i in np.arange(0, y.shape[0]):
                new_data = y[i, :]
                lines_multi_data[i].set_ydata(new_data)
                lines_multi_data[i].set_alpha(1)
                if i not in visible_channels:
                    lines_multi_data[i].set_alpha(0)
            trial_text.set_text("Template: " + str(current_template_index))
            plt.draw()
            update_marking_led()

    def on_prev():
        global current_template_index
        if current_template_index > 0:
            current_template_index -= 1
            template = np.argwhere(spike_templates == current_template_index)
            num_of_spikes_in_template = template.shape[0]
            while num_of_spikes_in_template == 0 and current_template_index > 0:
                current_template_index -= 1
                template = np.argwhere(spike_templates == current_template_index)
                num_of_spikes_in_template = template.shape[0]
            lines_multi_data = ax_multidim_data.get_lines()
            y = generate_data(current_template_index)
            visible_channels = get_visible_channels()
            for i in np.arange(0, y.shape[0]):
                new_data = y[i, :]
                lines_multi_data[i].set_ydata(new_data)
                lines_multi_data[i].set_alpha(1)
                if i not in visible_channels:
                    lines_multi_data[i].set_alpha(0)
            trial_text.set_text("Template: " + str(current_template_index))
            plt.draw()
            update_marking_led()

    def on_text_update():
        global current_template_index
        temp_template_index = int(textbox.text())
        if temp_template_index < data.shape[0] and temp_template_index > 0:
            current_template_index = temp_template_index
            lines_multi_data = ax_multidim_data.get_lines()
            y = generate_data(current_template_index)
            visible_channels = get_visible_channels()
            for i in np.arange(0, y.shape[0]):
                new_data = y[i, :]
                lines_multi_data[i].set_ydata(new_data)
                lines_multi_data[i].set_alpha(1)
                if i not in visible_channels:
                    lines_multi_data[i].set_alpha(0)
            trial_text.set_text("Template: " + str(current_template_index))
            plt.draw()
            update_marking_led()

    def on_keep():
        global current_template_index
        template_marking[current_template_index] = 1
        update_marking_led()
        np.save(os.path.join(base_folder, 'template_marking.npy'), template_marking)

    def on_delete():
        global current_template_index
        template_marking[current_template_index] = 0
        update_marking_led()
        np.save(os.path.join(base_folder, 'template_marking.npy'), template_marking)

    def on_slider_update():
        global visibility_threshold
        visibility_threshold = slider_threshold.value() / 10

    '''
    # visible channel selection based on user set threshold of size of amplitude of channel
    def get_visible_channels():
        global current_template_index
        global visibility_threshold
        amplitude = np.nanmax(templates[current_template_index, :, :]) - np.nanmin(templates[current_template_index, :, :])
        points_over_threshold = np.argwhere(templates[current_template_index, :, :] >
                                            (np.nanmax(templates[current_template_index, :, :]) -
                                             visibility_threshold * amplitude))
        channels_over_threshold = np.unique(points_over_threshold[:, 1])
        return channels_over_threshold
    '''

    def get_visible_channels():
        global current_template_index
        global visibility_threshold
        median = np.median(np.nanmin(templates[current_template_index, :, :], axis=0))
        std = np.std(np.nanmin(templates[current_template_index, :, :], axis=0))
        points_under_median = np.argwhere(templates[current_template_index, :, :] < (median - visibility_threshold * std))
        channels_over_threshold = np.unique(points_under_median[:, 1])
        print(len(channels_over_threshold))
        return channels_over_threshold

    def update_marking_led():
        global current_template_index
        if template_marking[current_template_index]:
            label_led_marking.setText('KEPT')
            label_led_marking.setPalette(kept_palette)
        else:
            label_led_marking.setText('DELETED')
            label_led_marking.setPalette(deleted_palette)

    def generate_data(template_index):
        if not np.any(data[template_index, :, :]):
            template = np.argwhere(spike_templates == template_index)
            num_of_spikes_in_template = template.shape[0]
            if num_of_spikes_in_template is not 0:
                y = np.zeros((num_of_channels, time_points * 2))
                print('Loading template ' + str(template_index) + ' with ' + str(num_of_spikes_in_template) +
                      ' spikes for the first time.')
                for i in np.arange(0, num_of_spikes_in_template):
                    y = y + data_raw_kilosorted[active_channel_map,
                            spike_times[template[i, 0]][0] - time_points: spike_times[template[i, 0]][0] + time_points]
                y = y / num_of_spikes_in_template
                data[template_index, :, :] = y
                np.save(os.path.join(base_folder, 'avg_spike_template.npy'), data)
            else:
                print('Template ' + str(template_index) + ' has no spikes.')
                y = np.squeeze(data[template_index, :, :])
        else:
            print('Loading template ' + str(template_index))
            y = np.squeeze(data[template_index, :, :])
        return y

    # ---------------------------------------------------

    # ------ Generating the Matplotlib Figure -----------
    starting_data = generate_data(0)

    fig = plt.figure(figure_id)

    time_axis = np.arange(-time_points, time_points, 1/freq)
    ax_multidim_data = pf.plot_avg_time_locked_data(starting_data, time_axis, subplot=None, timeToPlot=timeToPlot,
                                                    remove_channels=None, figure=fig)

    multidim_data_ax_offset = 0.07
    multidim_data_ax_height_change = multidim_data_ax_offset

    pos_multi = ax_multidim_data.get_position()
    new_pos_multi = [pos_multi.x0, pos_multi.y0 + multidim_data_ax_offset, pos_multi.width,
                     pos_multi.height - multidim_data_ax_height_change]
    ax_multidim_data.set_position(new_pos_multi)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    trial_text = plt.figtext(0.85, 0.85, "Template: " + str(0), ha="right", va="top", size=20, bbox=bbox_props)
    # ---------------------------------------------------

    # ----- The Qt Stuff --------------------------------
    root = fig.canvas.manager.window
    panel = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(panel)
    grid.setColumnStretch(1, 6)
    grid.setColumnStretch(2, 6)

    button_keep = QtWidgets.QPushButton('Keep')
    grid.addWidget(button_keep, 0, 0)
    button_keep.pressed.connect(on_keep)

    button_delete = QtWidgets.QPushButton('Delete')
    grid.addWidget(button_delete, 0, 1)
    button_delete.pressed.connect(on_delete)

    label_text_marking = QtWidgets.QLabel('Template is marked as :')
    grid.addWidget(label_text_marking, 0, 2)

    label_led_marking = QtWidgets.QLabel('DELETED')
    grid.addWidget(label_led_marking, 0, 3)
    label_led_marking.setAutoFillBackground(True)
    kept_palette = QtGui.QPalette()
    kept_palette.setColor(QtGui.QPalette.Background, Qt.green)
    deleted_palette = QtGui.QPalette()
    deleted_palette.setColor(QtGui.QPalette.Background, Qt.red)
    label_led_marking.setPalette(deleted_palette)

    button_previous = QtWidgets.QPushButton('Previous')
    grid.addWidget(button_previous, 0, 4)
    button_previous.pressed.connect(on_prev)

    button_next = QtWidgets.QPushButton('Next')
    grid.addWidget(button_next, 0, 5)
    button_next.pressed.connect(on_next)

    textbox = QtWidgets.QLineEdit(parent=panel)
    grid.addWidget(textbox, 1, 0)
    textbox.returnPressed.connect(on_text_update)

    slider_threshold = QtWidgets.QSlider(Qt.Horizontal)
    slider_threshold.setMinimum(1)
    slider_threshold.setMaximum(40)
    slider_threshold.setTickInterval(1)
    slider_threshold.setTickPosition(QtWidgets.QSlider.TicksBelow)
    slider_threshold.setSingleStep(1)
    slider_threshold.setValue(int(visibility_threshold*10))
    grid.addWidget(slider_threshold, 1, 1, 1, 5)
    slider_threshold.valueChanged.connect(on_slider_update)

    panel.setLayout(grid)

    dock = QtWidgets.QDockWidget("Controls", root)
    root.addDockWidget(Qt.BottomDockWidgetArea, dock)
    dock.setWidget(panel)


def generate_data_cube(base_folder, binary_data_filename, time_points=100):


    channel_map = np.load(os.path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(os.path.join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(os.path.join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    data_raw = np.memmap(os.path.join(base_folder, binary_data_filename),
                         dtype=np.int16, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_kilosorted = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

    spike_times = np.load(os.path.join(base_folder, 'spike_times.npy')).astype(np.int)

    num_of_channels = active_channel_map.size

    data = np.zeros((number_of_templates, num_of_channels, time_points * 2))

    for template_index in np.arange(0, number_of_templates):
        template = np.argwhere(spike_templates == template_index)
        num_of_spikes_in_template = template.shape[0]
        if num_of_spikes_in_template is not 0:
            y = np.zeros((num_of_channels, time_points * 2))
            div = num_of_spikes_in_template
            for i in np.arange(0, num_of_spikes_in_template):
                if data_raw_kilosorted[active_channel_map,
                        spike_times[template[i, 0]][0] - time_points: spike_times[template[i, 0]][0] + time_points].shape[1] < time_points * 2:
                    div -= 1
                else:
                    y = y + data_raw_kilosorted[active_channel_map,
                        spike_times[template[i, 0]][0] - time_points: spike_times[template[i, 0]][0] + time_points]
            y = y / div
            data[template_index, :, :] = y
            print('Added template ' + str(template_index) + ' with ' + str(num_of_spikes_in_template) + ' spikes')

    np.save(os.path.join(base_folder, 'avg_spike_template.npy'), data)




