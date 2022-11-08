

from os.path import join
import numpy as np
from spikesorting_tsne import io_with_cpp as tsne_io

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"

# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs

base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
data_folder = join(base_data_folder, 'FiringRateDataPrep')
save_data_folder = join(base_data_folder, 'Data', 'PCA_and_Tsne', 'data_samplesEvery2Frames_5secslong_tsneOfFrame_tsneOfBrain')

behaviour_tsne_results_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\Tsne\Video\CroppedVideo_100ms_Top100PCs'
brain_tsne_results_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\Tsne\SpikingVectors\All_spikes_100msbin_count_top40PCs_6Kiters'

num_of_frames = 32000


# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="Create dataset">

brain_tsne = tsne_io.load_tsne_result(brain_tsne_results_folder)
behaviour_tsne = tsne_io.load_tsne_result(behaviour_tsne_results_folder)


def sample_data(frames_per_packet, batch_size, start_frame_for_period=0, batch_step=1):

    X_buffer = np.empty(shape=(batch_size, 2*frames_per_packet + 2))
    #X_buffer = np.empty(shape=(batch_size, 2 * frames_per_packet))
    Y_buffer = np.empty(shape=(batch_size, 2))

    for i in np.arange(start_frame_for_period, batch_size, batch_step):

        tsne_behaviour_start = behaviour_tsne[i-600]
        tsne_behaviour_end = behaviour_tsne[i]
        brain = brain_tsne[i-frames_per_packet:i, :].flatten()

        X_buffer[i, :] = np.concatenate((brain, tsne_behaviour_start))
        #X_buffer[i, :] = brain
        Y_buffer[i, :] = tsne_behaviour_end

    np.save(join(save_data_folder, 'X_buffer_120b_600i.npy'), X_buffer)
    np.save(join(save_data_folder, 'Y_buffer_120b_600i.npy'), Y_buffer)

    return X_buffer, Y_buffer

frames_per_packet = int(10 * 120)
batch_step = 1
batch_size = num_of_frames

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
