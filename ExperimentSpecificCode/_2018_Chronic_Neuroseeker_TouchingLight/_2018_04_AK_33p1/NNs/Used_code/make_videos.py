

from os.path import join
import numpy as np
import cv2
import progressbar
import matplotlib.pyplot as plt

#   File names
base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
data_folder = join(base_data_folder, 'Data')
data_prep_folder = join(base_data_folder, 'FiringRateDataPrep')

spiky_model_file = join(data_folder, 'spikes_latest_model_first3minutes_half_size.h5')
picture_model_file = join(data_folder, 'pictures_latest_model_half_size.h5')

save_data_file = join(data_folder, 'data_first3minutes_5secs_half_size_res.npz')
full_firing_matrix_file = join(data_folder, 'full_matrix.npy')

subsampled_read_video = join(base_data_folder, 'SubsampledVideo', 'Video_undistrorted_150x112_120fps.mp4')

saved_video_resolution = (75, 56)

#   Load data

spike_info = np.load(join(data_prep_folder, 'spike_info_after_cortex_sorting.df'), allow_pickle=True)
templates = np.unique(spike_info['template_after_sorting'])

cap = cv2.VideoCapture(subsampled_read_video)

dpi = 100
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

keras_in = False
try:
    import keras
    spiky_model = keras.models.load_model(spiky_model_file)
    keras_in = True
except ModuleNotFoundError:
    pass

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
full_firing_matrix = np.memmap(full_firing_matrix_file,
                        dtype=np.int16, mode='r', shape=(len(templates), total_video_frames)).T

full_firing_matrix_random = np.random.random_integers(full_firing_matrix.min(), full_firing_matrix.max(), full_firing_matrix.shape)

pred_image_frame_jump = 600
final_frame_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - pred_image_frame_jump - 1)

results_panel_resolution = (int(saved_video_resolution[0] * 4), int(saved_video_resolution[1] * 4))


def process_frame(frame_data):
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    frame_data = cv2.resize(frame_data, saved_video_resolution, interpolation=cv2.INTER_AREA)
    return frame_data


def reverse_process_frame(frame_data):
    frame_data = cv2.resize(frame_data, results_panel_resolution, interpolation=cv2.INTER_AREA)
    return frame_data


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


final_frame_index= 120*360 + pred_image_frame_jump


def make_2_panel_video():

    write_to_video = join(data_folder, 'results_video_1hour_halfsize_5secs_BrainOnly_fromFirst3minutes.avi')
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

make_2_panel_video()