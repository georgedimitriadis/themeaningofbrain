

from os.path import join
import numpy as np
import cv2
import progressbar
import matplotlib.pyplot as plt

#   File names

video_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
subsampled_read_video = join(video_folder, 'SubsampledVideo', 'Video_undistrorted_150x112_120fps.mp4')


#   Load data

cap = cv2.VideoCapture(subsampled_read_video)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
saved_video_resolution = (75, 56)

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results_panel_resolution = (int(saved_video_resolution[0] * 4), int(saved_video_resolution[1] * 4))

keras_in = False
try:
    import keras
#    spiky_model = keras.models.load_model(spiky_model_file)
#    print(spiky_model)
    keras_in = True
except ModuleNotFoundError:
    pass

print('HAS FOUND KERAS: {}'.format(keras_in))



def process_frame(frame_data):
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    frame_data = cv2.resize(frame_data, saved_video_resolution, interpolation=cv2.INTER_AREA)
    return frame_data


def reverse_process_frame(frame_data):
    frame_data = cv2.resize(frame_data, results_panel_resolution, interpolation=cv2.INTER_AREA)
    return frame_data

'''
def make_4_panel_video():

    write_to_video = join(data_folder, 'results_video_1hour_halfsize_5secs_BrainAndPics.avi')
    results_video_resolution = ((2 * results_panel_resolution[1]) + 20, (2 * results_panel_resolution[0]) + 20)
    cap_results = cv2.VideoWriter(write_to_video, fourcc, 120.0,
                                  (results_video_resolution[1], results_video_resolution[0]))

    for f in np.arange(final_frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, starting_image = cap.read()
        starting_image = process_frame(starting_image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f + pred_image_frame_jump)
        _, ending_image = cap.read()
        ending_image = process_frame(ending_image)
        X = full_firing_matrix[f:f+pred_image_frame_jump, :]

        spikes = X.reshape(1, X.shape[0], X.shape[1], 1)
        si = starting_image.reshape(1, 1, starting_image.shape[0], starting_image.shape[1])
        si = si / 255.0
        if keras_in:
            pred_spiky_image = np.squeeze(spiky_model.predict([spikes, si])) * 255
            picture_model = keras.models.load_model(picture_model_file)
            pred_pics_image = np.squeeze(picture_model.predict(si)) * 255
        else:
            pred_spiky_image = np.ones(saved_video_resolution) * 200
            pred_pics_image = np.ones(saved_video_resolution) * 200

        starting_image = reverse_process_frame(starting_image)
        ending_image = reverse_process_frame(ending_image)

        buf = np.zeros(results_video_resolution)
        x_pixels = results_panel_resolution[1]
        y_pixels = results_panel_resolution[0]
        buf[:x_pixels, :y_pixels] = starting_image
        buf[-x_pixels:, :y_pixels] = reverse_process_frame(pred_pics_image)
        buf[:x_pixels, -y_pixels:] = ending_image
        buf[-x_pixels:, -y_pixels:] = reverse_process_frame(pred_spiky_image)

        cv2.imwrite(join(data_folder, 'temp.png'), buf)
        temp = cv2.imread(join(data_folder, 'temp.png'))
        cap_results.write(temp)

        if f % 1000 == 0:
            print(f)



def make_2_panel_video():

    write_to_video = join(data_folder, 'results_video_1hour_halfsize_2secs_BrainOnly_from25Krandom.avi')
    results_video_resolution = (results_panel_resolution[1], (2 * results_panel_resolution[0]) + 20)
    cap_results = cv2.VideoWriter(write_to_video, fourcc, 120.0,
                                  (results_video_resolution[1], results_video_resolution[0]))

    bar = progressbar.bar.ProgressBar(min_value=pred_image_frame_jump, max_value=final_frame_index)
    for f in np.arange(pred_image_frame_jump, final_frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, image = cap.read()
        image = process_frame(image)
        X = full_firing_matrix[f - pred_image_frame_jump: f, :]
        spikes = X.reshape(1, X.shape[0], X.shape[1], 1)
        if keras_in:
            pred_spiky_image = np.squeeze(spiky_model.predict([spikes])) * 255
        else:
            pred_spiky_image = np.ones(saved_video_resolution) * 200

        image = reverse_process_frame(image)

        buf = np.zeros(results_video_resolution)
        x_pixels = results_panel_resolution[1]
        y_pixels = results_panel_resolution[0]
        buf[:x_pixels, :y_pixels] = image
        buf[:x_pixels, -y_pixels:] = reverse_process_frame(pred_spiky_image)

        cv2.imwrite(join(data_folder, 'temp.png'), buf)
        temp = cv2.imread(join(data_folder, 'temp.png'))
        cap_results.write(temp)

        bar.update(f)
        #if f % 1000 == 0:
        #    print('Done {} from {}'.format(f, final_frame_index - pred_image_frame_jump))
'''

def make_single_video(video_file, model_file=None, firing_matrix_file=None, pred_image_frame_jump=None,
                      starting_frame_index=None, final_frame_index=None):

    if model_file is not None and not keras_in:
        return



def make_n_x_m_panels_video(video_folder, video_file, model_files, firing_matrices_files, pred_image_frame_jumps,
                            arrangement, starting_frame_index=None, final_frame_index=None):
    """

    :param video_folder: The folder where the video will be saved (and the temp.png required)

    :param video_file: The name of the video to be saved in the video_folder

    :param model_files: A list of full file names of all the models to use

    :param firing_matrices_files: A list of full file names of the corresponding firing matrices

    :param pred_image_frame_jumps: A list of the lengths of the windows used for each model
        (how many frames worth of data each model works with)

    :param arrangement: A list of two element tuples showing the position of each vide. This list needs to be one
        longer than the model_files list because the first tuple should show the position of the raw video

    :param starting_frame_index: The frame index of the movie to start from. The function won't check for errors
        so it must be a positive number not larger than the total frames in the movie minus the largest window of frames.
        Default: None -> Means the beginning of the movie (given the largest window of frames

    :param final_frame_index: The frame index of the movie to end at. It must be a positive number larger than the
        starting_frame_index plus the largest frames window and smaller than the whole movie.
        Default: None -> The end of the movie
    """

    if not keras_in:
        return

    video_file = join(video_folder, video_file)

    gap = 10

    x_coords = np.array(arrangement)[:, 0]
    y_coords = np.array(arrangement)[:, 1]

    number_of_columns = x_coords.max() + 1
    number_of_rows = y_coords.max() + 1

    results_video_resolution = (number_of_rows * results_panel_resolution[1] + (number_of_rows-1) * gap,
                                (number_of_columns * results_panel_resolution[0]) + (number_of_columns - 1) * gap)

    cap_results = cv2.VideoWriter(video_file, fourcc, 120.0,
                                  (results_video_resolution[1], results_video_resolution[0]))

    pred_image_frame_jump_max = np.max(pred_image_frame_jumps)
    if starting_frame_index is None:
        starting_frame_index = pred_image_frame_jump_max
    if final_frame_index is None:
        final_frame_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - pred_image_frame_jump_max - 1)

    models = []
    for model_file in model_files:
        models.append(keras.models.load_model(model_file))

    firing_matrices = []
    for i, firing_matrix_file in enumerate(firing_matrices_files):
        firing_matrices.append(np.memmap(firing_matrix_file, dtype=np.int16, mode='r',
                                         shape=(models[i].input_shape[2], total_video_frames)).T)

    bar = progressbar.bar.ProgressBar(min_value=starting_frame_index, max_value=final_frame_index)
    for f in np.arange(starting_frame_index, final_frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, image = cap.read()
        image = process_frame(image)

        image = reverse_process_frame(image)

        buf = np.zeros(results_video_resolution)

        # Make the pic for the real image
        position = arrangement[0]  # The first tuple in arrangement is the position of the real image
        x_position = position[0]
        y_position = position[1]

        x_begin_pixels = x_position * results_panel_resolution[1] + x_position * gap
        x_end_pixels = x_begin_pixels + results_panel_resolution[1]
        y_begin_pixels = y_position * results_panel_resolution[0] + y_position * gap
        y_end_pixels = y_begin_pixels + results_panel_resolution[0]

        buf[x_begin_pixels:x_end_pixels, y_begin_pixels:y_end_pixels] = image

        # Make the pics for the predicted images
        for i, model in enumerate(models):
            pred_image_frame_jump = pred_image_frame_jumps[i]
            firing_matrix = firing_matrices[i]
            X = firing_matrix[f - pred_image_frame_jump: f, :]
            spikes = X.reshape(1, X.shape[0], X.shape[1], 1)
            predicted_image = np.squeeze(model.predict([spikes])) * 255

            position = arrangement[i + 1]
            x_position = position[0]
            y_position = position[1]

            x_begin_pixels = x_position * results_panel_resolution[1] + x_position * gap
            x_end_pixels = x_begin_pixels + results_panel_resolution[1]
            y_begin_pixels = y_position * results_panel_resolution[0] + y_position * gap
            y_end_pixels = y_begin_pixels + results_panel_resolution[0]

            buf[x_begin_pixels:x_end_pixels, y_begin_pixels:y_end_pixels] = reverse_process_frame(predicted_image)

        cv2.imwrite(join(video_folder, 'temp.png'), buf)
        temp = cv2.imread(join(video_folder, 'temp.png'))
        cap_results.write(temp)

        bar.update(f)



video_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs\Comparative_results'
video_file = 'results_video_FullVideo_halfsize_[2sNS_2sL_2sS]_BrainOnly_from25Krandom.avi'

model_files = [r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs\Data\randomly_selected_frames_brain_only\spikes_latest_model_fullvideo_25Krandom_2secs.h5',
               r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs\Data\randomly_selected_frames_brain_only\spikes_latest_model_fullvideo_25Krandom_2secs.h5',
               r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs\Data\randomly_selected_frames_brain_only\spikes_latest_model_fullvideo_25Krandom_2secs.h5']

firing_matrices_files = [r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs\FiringRateDataPrep\full_firing_matrix.npy',
               r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs\FiringRateDataPrep\full_firing_matrix.npy',
               r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs\FiringRateDataPrep\full_firing_matrix.npy']

pred_image_frame_jumps = [240, 240, 240]
arrangement = [(0, 0), (0, 1), (1, 0), (1, 1)]


# final_frame_index= 120*360 + pred_image_frame_jump  # Do that for a shorter movie

make_n_x_m_panels_video(video_folder, video_file, model_files, firing_matrices_files, pred_image_frame_jumps, arrangement,
                            starting_frame_index=None, final_frame_index=None)

