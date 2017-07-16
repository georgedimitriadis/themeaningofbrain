
import numpy as np

def create_data_cube_from_raw_extra_data(raw_extracellular_data, data_cube_filename,
                                         num_of_points_in_spike_trig, cube_type, extra_spike_times,
                                         num_of_electrodes=None, used_electrodes=None,
                                         num_of_points_for_baseline=None):
    import os.path as path
    if path.isfile(data_cube_filename):
        import os
        os.remove(data_cube_filename)

    num_of_spikes = len(extra_spike_times)
    if used_electrodes is None and num_of_electrodes is not None:
        used_electrodes = np.arange(num_of_electrodes)
    elif used_electrodes is not None:
        num_of_electrodes = used_electrodes.shape[0]
    else:
        print('Please provide either a number of electrodes or a list of electrodes')
        return
    shape_of_spike_trig_avg = ((num_of_electrodes,
                                num_of_points_in_spike_trig,
                                num_of_spikes))

    data_cube = np.memmap(data_cube_filename,
                          dtype=cube_type,
                          mode='w+',
                          shape=shape_of_spike_trig_avg)
    for spike in np.arange(0, num_of_spikes):
        trigger_point = extra_spike_times[spike]
        start_point = int(trigger_point - num_of_points_in_spike_trig / 2)
        if start_point < 0:
            break
        end_point = int(trigger_point + num_of_points_in_spike_trig / 2)
        if end_point > raw_extracellular_data.shape[1]:
            break
        temp = raw_extracellular_data[used_electrodes, start_point:end_point]
        if num_of_points_for_baseline is not None:
            baseline = np.mean(temp[:, [0, num_of_points_for_baseline]], 1)
            temp = (temp.T - baseline.T).T
        data_cube[:, :, spike] = temp.astype(cube_type)
        if spike % 1000 == 0:
            print('Done ' + str(spike) + ' spikes')
        del temp
    del raw_extracellular_data
    del baseline
    del data_cube

    cut_extracellular_data = load_extracellular_data_cube(data_cube_filename, cube_type, shape_of_spike_trig_avg)

    return cut_extracellular_data



def load_extracellular_data_cube(data_cube_filename, cube_type,
                                 shape_of_spike_trig_avg):
    cut_extracellular_data = np.memmap(data_cube_filename,
                                       dtype=cube_type,
                                       mode='r',
                                       shape=shape_of_spike_trig_avg)
    return cut_extracellular_data
