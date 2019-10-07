
import numpy as np
from os.path import join

data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs\Data'

spike_matrix_file = join(data_folder, 'full_matrix.npy')

spike_atrix_dtype = np.int16

number_of_templates = 838

video_file = join(data_folder, 'Video_undistrorted_150x112_120fps.mp4')

save_data_file = join(data_folder, 'data_sequential_chuncked.npz')
spiky_model_file = join(data_folder, 'spiky_model.h5')
spiky_checkpoint_file = join(data_folder, 'spiky_best_current_model.h5')
picture_model_file = join(data_folder, 'picture_model.h5')
picture_checkpoint_file = join(data_folder, 'picture_best_current_model.h5')

frames_per_packet = 360

