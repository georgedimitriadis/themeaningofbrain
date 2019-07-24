
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

global marker_dots

'''
# Look at the trajectory with an arrow at the end for the current speeds
frames_to_average = int(120 * 0.5 * 0.5)
def update_trajectory(frame, figure):
    figure.clear()
    traj_x = body_positions[:frame, 0]
    traj_y = body_positions[:frame, 1]
    speed_x = np.nanmean(body_velocities[frame-frames_to_average:frame+frames_to_average, 0]) * frames_to_average
    speed_y = np.nanmean(body_velocities[frame-frames_to_average:frame+frames_to_average, 1]) * frames_to_average
    ax = figure.add_subplot(111)
    ax.plot(traj_x, traj_y)
    ax.arrow(traj_x[-1], traj_y[-1], speed_x, speed_y, head_width=0.5, head_length=0.5)

    return speed_x


def update_trajectory_for_video(frame):
    traj[:, :, :] = 0
    bp = body_positions.astype(int)
    bp[:, 1] = 640 - bp[:, 1]
    frames_to_show = 30
    marker_size = 3

    if frame < frames_to_show:
        traj[bp[:frame, 1], bp[:frame, 0], :] = 255
    else:
        traj[bp[frame - frames_to_show:frame, 1], bp[frame - frames_to_show:frame, 0], :] = 255
    traj[bp[frame, 1]-marker_size:bp[frame, 1]+marker_size, bp[frame, 0]-marker_size:bp[frame, 0]+marker_size, :] = 255

    markers = updated_body_markers_positions.loc[frame]
    neck = markers[:2].values.astype(int)
    body = markers[2:4].values.astype(int)
    tail = markers[4:6].values.astype(int)
    traj[neck[1]-marker_size:neck[1]+marker_size, neck[0]-marker_size:neck[0]+marker_size, 0] = 255
    traj[neck[1]-marker_size:neck[1]+marker_size, neck[0]-marker_size:neck[0]+marker_size, 3] = 255
    traj[body[1]-marker_size:body[1]+marker_size, body[0]-marker_size:body[0]+marker_size, 2] = 255
    traj[body[1]-marker_size:body[1]+marker_size, body[0]-marker_size:body[0]+marker_size, 3] = 255
    traj[tail[1]-marker_size:tail[1]+marker_size, tail[0]-marker_size:tail[0]+marker_size, 1] = 255
    traj[tail[1]-marker_size:tail[1]+marker_size, tail[0]-marker_size:tail[0]+marker_size, 3] = 255

    return output

'''


def update_markers_for_video(frame, position_only_markers, marker_size):
    """
    This is a function that should be run by a transform (from the interactive_programming_in_python) and it will
    generate squares where the positions of the markers dataframe passed to it are. This marker_dots variable (picture)
    that needs to be global can then be superimposed on a video.

    It is meant to be used like:

    global marker_dots
    marker_dots = np.zeros((640, 640, 4))
    dlc_plotting.marker_dots = marker_dots
    args = [position_only_markers, marker_size]
    transform.connect_repl_var(globals(), 'frame', 'update_head_markers_for_video', 'output', 'args')
    sv.image_sequence(globals(), 'frame', 'video_file', 'marker_dots')

    :param frame: The frame to plot
    :param position_only_markers: the markers dataframe that has only positions in it (for each marker an x and a y)
    :return: None
    """
    global marker_dots
    marker_dots[:, :, :] = 0

    colormap = cm.jet
    markers = position_only_markers.loc[frame]
    number_of_parts = len(np.unique(position_only_markers.columns.get_level_values(1)))
    for index in np.arange(0, 2 * number_of_parts, 2):
        data = markers[index:index + 2].values.astype(int)
        color = colormap(index / (2 * number_of_parts))
        if ~np.isnan(data[0]) and ~np.isnan(data[1]):
            marker_dots[data[1] - marker_size:data[1] + marker_size, data[0] - marker_size:data[0] + marker_size, 3] = 255
            for c in np.arange(0, 3):
                marker_dots[data[1] - marker_size:data[1] + marker_size, data[0] - marker_size:data[0] + marker_size, c] = \
                    color[c] * 255

    return None


