


from os.path import join
import numpy as np

import cv2

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"

# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs

base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs'
data_folder = join(base_data_folder, 'FiringRateDataPrep')
save_data_folder = join(base_data_folder, 'Data', 'RandomisedInput')

video_events_file = join(data_folder, 'Video.pkl')
sampling_freq = 20000
time_points_per_frame = 166

#spike_info = np.load(join(data_folder, 'spike_info_after_cortex_sorting.df'), allow_pickle=True) # For the full
spike_info = np.load(join(data_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
templates = np.unique(spike_info['template_after_sorting'])
video_events = np.load(join(data_folder, 'Video.pkl'), allow_pickle=True)
video_times = video_events.values

num_of_neurons = len(templates)
num_of_neurons_full = 838
num_of_frames = len(video_times)

#del spike_info
#del templates
#del video_events

video_folder = join(r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs', 'SubsampledVideo')

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Create dataset">
cap = cv2.VideoCapture(join(video_folder, 'Video_undistrorted_150x112_120fps.mp4'))

# For Neuropixel
matrix_name = 'full_matrix_with_extra_poisson_data.npy'
dtype = np.int32

# For Full
#matrix_name = 'full_firing_matrix.npy'
#dtype = np.int16

full_matrix = np.memmap(join(data_folder, matrix_name),
                        dtype=dtype, mode='r', shape=(num_of_frames, num_of_neurons_full))

#_lambda = np.mean(fullmatrix)
#full_matrix = np.random.poisson(_lambda, fullmatrix.shape)

def sample_data(filename_to_save, frames_per_packet, batch_size, start_frame_for_period=None, batch_step=1):

    import progressbar

    X_0 = []
    r = []
    Y = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = progressbar.bar.ProgressBar(max_value=batch_size)
    for i in range(batch_size):
        X_current_buffer = []
        Y_current_buffer = []

        if start_frame_for_period == None:
            r_int = np.random.randint(total - frames_per_packet )
        else:
            r_int = start_frame_for_period + i * batch_step
        r.append(r_int)

        for j in range(frames_per_packet):

            x = full_matrix[r_int + j]

            '''
            num_of_zeros = len(np.where(x==0))
            max = x.max()
            if max == 0:
                max = 1
            x = np.random.randint(0, high=max, size=len(x))
            x[np.random.choice(np.arange(len(x)), num_of_zeros)] = 0
            '''
            X_current_buffer.append(np.array(x, dtype=np.float32, copy=False))
            if j == frames_per_packet-1 or j == 0:
                if j == 0:
                    dt = 0
                    p = 0
                else:
                    dt = frames_per_packet
                    p = 1
                cap.set(1, r_int + dt)
                ret, frame = cap.read()
                y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                y = cv2.resize(y, (150 // 2, 112 // 2), interpolation=cv2.INTER_AREA)
                Y_current_buffer.append(np.array(y, dtype=np.float32, copy=False))

        X_0.append(X_current_buffer)
        Y.append(Y_current_buffer)
        bar.update(i)

    X_0 = np.array(X_0, dtype=np.float32, copy=False)
    r = np.array(r, dtype=np.float32, copy=False)
    Y = np.array(Y, dtype=np.float32, copy=False)

    np.savez(join(save_data_folder, filename_to_save), r=r,  X=X_0, Y=Y)


# For random sampling
sample_data("data_extra_poisson_25000randompoints_2secslong_halfsizeres.npz", 2*120, 25000,
            start_frame_for_period=None, batch_step=1)


# </editor-fold>


# Create the randomised full matrix
fm = np.copy(full_matrix)
np.random.shuffle(fm)
for i, f in enumerate(fm):
    np.random.shuffle(f)
    fm[i] = f

np.save(join(data_folder, 'full_noise.npy'), fm)

# Try again
fullmatrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(num_of_neurons, num_of_frames)).T

full_matrix = np.copy(fullmatrix)
np.random.shuffle(full_matrix.flat)
np.random.shuffle(full_matrix.flat)
np.random.shuffle(full_matrix.flat)
np.save(join(data_folder, 'fully_shuffled_x3_matrix.npy'), full_matrix)


# Use the fully_shuffled_matrix to add shuffled data to the Neuropixel Simulations
full_matrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(num_of_neurons, num_of_frames)).T
noise = np.load(join(r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs', 'FiringRateDataPrep', 'fully_shuffled_matrix.npy'))
full_matrix_with_extra_shuffled_data = np.empty(noise.shape)
full_matrix_with_extra_shuffled_data[:, :full_matrix.shape[1]] = np.copy(full_matrix)
full_matrix_with_extra_shuffled_data[:, full_matrix.shape[1]:] = np.copy(noise[:, full_matrix.shape[1]:])
np.save(join(data_folder, 'full_matrix_with_extra_shuffled_data.npy'), full_matrix_with_extra_shuffled_data)


# Use the poisson random data to add data to the Neuropixel Simulations

fullmatrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(num_of_neurons, num_of_frames)).T
_lambda = np.mean(fullmatrix)
poisson_data = np.random.poisson(_lambda, (fullmatrix.shape[0], num_of_neurons_full-num_of_neurons))
full_matrix_with_extra_poisson = np.concatenate((fullmatrix, poisson_data), axis=1)
np.save(join(data_folder, 'full_matrix_with_extra_poisson_data.npy'), full_matrix_with_extra_poisson)
