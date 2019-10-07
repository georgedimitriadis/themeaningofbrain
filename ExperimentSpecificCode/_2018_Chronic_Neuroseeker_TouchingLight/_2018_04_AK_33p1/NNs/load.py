
import cv2
import numpy as np

try:
    from . import constants as const
except ImportError:
    from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1.NNs \
    import constants as const


cap = cv2.VideoCapture(const.video_file)
num_of_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

full_matrix = np.memmap(const.spike_matrix_file, dtype=const.spike_atrix_dtype, mode='r',
                        shape=(const.number_of_templates, num_of_video_frames)).T


def process_frame(frame_data):
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    frame_data = cv2.resize(frame_data, (150 // 4, 112 // 4), interpolation=cv2.INTER_AREA)
    return frame_data


def generate_chunked_dataset(frame_regions, examples_size):
    import progressbar

    X_chunk = []
    Y_chunk = []
    r_chunk = []

    frames_per_packet = const.frames_per_packet

    bar = progressbar.ProgressBar(max_value=examples_size)
    for i in range(examples_size):
        X_current_buffer = []
        Y_current_buffer = []

        r_region_index = np.random.choice(np.arange(len(frame_regions)))
        from_frame = frame_regions[r_region_index][0]
        to_frame = frame_regions[r_region_index][1]
        r_int = np.random.randint(from_frame, to_frame - frames_per_packet - 1)
        r_chunk.append(r_int)

        for j in range(frames_per_packet):

            x = full_matrix[r_int + j]
            X_current_buffer.append(np.array(x, dtype=np.float32, copy=False))

            if (frames_per_packet - 1 == j or j == 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, r_int + j)
                ret, frame = cap.read()
                y = process_frame(frame)
                Y_current_buffer.append(np.array(y, dtype=np.float32, copy=False))
        X_chunk.append(X_current_buffer)
        Y_chunk.append(Y_current_buffer)
        if i%1000 == 0:
            bar.update(i)

    X_chunk = np.array(X_chunk, dtype=np.float32, copy=False)
    Y_chunk = np.array(Y_chunk, dtype=np.float32, copy=False)
    r_chunk = np.array(r_chunk, dtype=np.float32, copy=False)

    return X_chunk, Y_chunk, r_chunk


def generate_full_train_test_dataset(num_of_chunks=3, example_size_train=25000):
    """
    Cut the frames in num_of_mini_batches parts. Separate each part in train (90%) and test (10%). From each train and
    test part pick randomly mini_batch_size number of frames and make 3 secs examples from each.
    That means that the train and test sets are spread throughout the data and that the test sets have enough
    continuity to be used later to make a video

    :param num_of_chunks: number of regions of a train and a test mini_batch
    :param example_size_train: the number of examples pulled out of a region
    :return: Nothing
    """

    test_train_ratio = 0.1
    chunk_percentage = 1 / num_of_chunks
    train_chunk_percentage = chunk_percentage * (1 - test_train_ratio)
    test_chunk_percentage = chunk_percentage * test_train_ratio

    train_chunk_num_of_frames = int(train_chunk_percentage * num_of_video_frames)
    test_chunk_num_of_frames = int(test_chunk_percentage * num_of_video_frames)

    train_start_end_frames = [[0, train_chunk_num_of_frames]]
    test_start_end_frames = [[train_chunk_num_of_frames + 1,
                              train_chunk_num_of_frames + test_chunk_num_of_frames]]
    for i in np.arange(1, num_of_chunks):
        train_start_end_frames.append([test_start_end_frames[i-1][1] + 1,
                                       test_start_end_frames[i - 1][1] + 1 + train_chunk_num_of_frames])
        test_start_end_frames.append([train_start_end_frames[i][1] + 1,
                                      train_start_end_frames[i][1] + 1 + test_chunk_num_of_frames])

    X_train, Y_train, r_train = generate_chunked_dataset(train_start_end_frames, example_size_train)
    print('Finished making TRAIN set')
    print('----')
    X_test, Y_test, r_test = generate_chunked_dataset(test_start_end_frames,
                                                      int(example_size_train * test_train_ratio))
    print('Finished making TEST set')
    print('----')
    '''
    for i in np.arange(num_of_chunks):
        print(train_start_end_frames[i])
        X_m_tr, Y_m_tr, r_m_tr = generate_chunked_dataset(train_start_end_frames[i][0], train_start_end_frames[i][1],
                                                          chunk_size_train)
        print('')
        print('Finished making TRAIN mini batch number {}'.format(i))
        print('----')
        print(test_start_end_frames[i])
        X_m_te, Y_m_te, r_m_te = generate_chunked_dataset(test_start_end_frames[i][0], test_start_end_frames[i][1],
                                                          int(chunk_size_train * test_train_ratio))
        print('')
        print('Finished making TEST mini batch number {}'.format(i))
        print('---------')

        if i==0:
            X_train = X_m_tr
            Y_train = Y_m_tr
            r_train = r_m_tr
            X_test = X_m_te
            Y_test = Y_m_te
            r_test = r_m_te
        else:
            X_train = np.concatenate((X_train, X_m_tr), axis=0)
            Y_train = np.concatenate((Y_train, Y_m_tr), axis=0)
            r_train = np.concatenate((r_train, r_m_tr), axis=0)
            X_test = np.concatenate((X_test, X_m_te), axis=0)
            Y_test = np.concatenate((Y_test, Y_m_te), axis=0)
            r_test = np.concatenate((r_test, r_m_te), axis=0)
    '''

    Y_train = Y_train / 255.0
    Y_test = Y_test / 255.0

    starting_images_train = Y_train[:, 0:1, :, :]
    ending_images_train = Y_train[:, 1, :, :]
    starting_images_test = Y_test[:, 0:1, :, :]
    ending_images_test = Y_test[:, 1, :, :]

    np.savez(const.save_data_file,
             X_train=X_train, starting_images_train=starting_images_train,
             ending_images_train=ending_images_train, r_train=r_train,
             X_test=X_test, starting_images_test=starting_images_test,
             ending_images_test=ending_images_test, r_test=r_test)

