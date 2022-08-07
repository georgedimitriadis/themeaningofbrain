

from os.path import join
import numpy as np
from spikesorting_tsne import io_with_cpp as tsne_io
from sklearn.decomposition import PCA
import progressbar

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"

# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs

base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs'
data_folder = join(base_data_folder, 'FiringRateDataPrep')
save_data_folder = join(base_data_folder, 'Data', 'PCA_and_Tsne', 'data_samplesEvery2Frames_5secslong_tsneOfFrame')

#spike_info = np.load(join(data_folder, 'spike_info_after_cortex_sorting.df'), allow_pickle=True) #spike_info_after_cortex_sorting.df for the Full NeuroSeeker probe
spike_info = np.load(join(data_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
templates = np.unique(spike_info['template_after_sorting'])
video_events = np.load(join(data_folder, 'Video.pkl'), allow_pickle=True)
video_times = video_events.values

num_of_neurons = len(templates)
num_of_frames = len(video_times)

del spike_info
del templates
del video_events

behaviour_tsne_results_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\Tsne\Video\CroppedVideo_100ms_Top100PCs'

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Create dataset">
full_matrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(num_of_neurons, num_of_frames)).T

behaviour_tsne = tsne_io.load_tsne_result(behaviour_tsne_results_folder)

number_of_top_primary_components_used = 40

def sample_data(frames_per_packet, batch_size, start_frame_for_period=0, batch_step=1):

    r = []
    X_buffer = np.memmap(join(save_data_folder, 'X_buffer.npy'), dtype=np.float32, mode='w+',
                         shape=(batch_size, number_of_top_primary_components_used * full_matrix.shape[1] + 2))
    Y_buffer = np.memmap(join(save_data_folder, 'Y_buffer.npy'), dtype=np.float32, mode='w+',
                         shape=(batch_size, 2))

    #prog_bar = progressbar.bar.ProgressBar(max_value=batch_size)
    for i in range(batch_size):

        r_int = start_frame_for_period + i * batch_step
        r.append(r_int)
        print(str(r_int))
        brain = full_matrix[r_int:r_int+frames_per_packet]

        pca = PCA(n_components=number_of_top_primary_components_used)

        pca = pca.fit(brain)

        tsne_start = behaviour_tsne[r_int//12]

        tsne_end = behaviour_tsne[(r_int + frames_per_packet)//12 - 1]

        X_buffer[i, :] = np.concatenate((pca.components_.flatten(), tsne_start))

        Y_buffer[i, :] = tsne_end

        #prog_bar.update(i)

    r = np.array(r, dtype=np.float32, copy=False)

    np.savez(join(save_data_folder, 'binary_headers.npz'), dtype=[np.float32],
             shape_X=[batch_size, number_of_top_primary_components_used * full_matrix.shape[1] + 2],
             shape_Y=[batch_size, 2],
             r=r)

    print('/nStart frame = {}, End frame = {}'.format(r[0], r[-1]))



frames_per_packet = 5 * 120
batch_step = 2
batch_size = num_of_frames // batch_step - 2 * frames_per_packet

# Data set that will allow TimeSeriesSplit (with n=10) with a 2 frame jump and 108K samples (so that a 1/10 chunk has 10K samples in it)
sample_data(frames_per_packet=frames_per_packet, batch_size=batch_size,
            start_frame_for_period=frames_per_packet, batch_step=batch_step)

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Visualise dataset">
if False:
    import slider as sl

    headers = np.load(join(save_data_folder, 'binary_headers.npz'), allow_pickle=True)
    X = np.memmap(join(save_data_folder, "X_buffer.npy"), dtype=headers['dtype'][0], shape=tuple(headers['shape_X']))
    Y = np.memmap(join(save_data_folder, "Y_buffer.npy"), dtype=headers['dtype'][0], shape=tuple(headers['shape_Y']))

    def show_X(f):
        a1.cla()
        a2.cla()
        a3.cla()

        d = X[f]
        d_bin = np.argwhere(d>0)
        a1.scatter(d_bin[:,0], d_bin[:,1],s=3)
        a1.set_title('Brain')

        im_before = Y[f, 0, :, :]
        a2.imshow(im_before)
        a2.set_title('Image Before')

        im_after = Y[f, 1, :, :]
        a3.imshow(im_after)
        a3.set_title('Image After')


    fig1 = plt.figure(0)
    a1 = fig1.add_subplot(111)

    fig2 = plt.figure(1)
    a2 = fig2.add_subplot(111)

    fig3 = plt.figure(2)
    a3 = fig3.add_subplot(111)

    out = None
    f=0

    sl.connect_repl_var(globals(),'f', 'out', 'show_X', slider_limits=[0, X.shape[0]-1])
# </editor-fold>
