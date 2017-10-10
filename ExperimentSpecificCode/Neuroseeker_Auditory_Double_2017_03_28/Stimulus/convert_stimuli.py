
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    combine stimulus information/synchronization
"""

import sys
import os.path as op
import numpy as np
import pickle
import glob
import scipy.io as sio

def convert_stim(data_path):

    # recording sampling rate was always 20 kHz
    samplerate = 20000.

    # paths
    stim_path = op.join(data_path, 'Stimuli')
    stimgen_path = op.join(stim_path, 'stimgen')
    wav_path = op.join(stim_path, 'wavefiles')

    # ----------------------------------
    # get sync TTL pulses
    # ----------------------------------

    # both probes have the same subdirectory structure
    sort_path = op.join('Analysis',
                        'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                        'Kilosort')

    TTL_probes = {}
    for probe in ['Angled', 'Vertical']:

        f = op.join(data_path, probe, sort_path,
                    'Exp2and3_2017_03_28T18_48_25_Sync_U16_LP3p5KHz.bin')

        sync_signal = np.fromfile(f, dtype='int16')
        sync_signal -= sync_signal.min()
        TTL_probes[probe] = np.where(np.diff(sync_signal) > 0)[0] + 1

    # ------------------------------------------------
    # get stimulus parameters and sync pulse timestamps
    # ------------------------------------------------

    stim_data = {}
    TTL_index = 0

    # ----------------------------------
    # silence
    # ----------------------------------
    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_18-53-49_Silence_StimGen.pickle'), 'rb') as f:
        stimgen = pickle.load(f)

    ts = {'Angled': [TTL_probes['Angled'][TTL_index]],
          'Vertical': [TTL_probes['Vertical'][TTL_index]]}
    stim_data['Silence'] = {'duration': stimgen.duration,
                            'timestamps': ts,
                            'samplerate': samplerate,
                            'events': [{'Onset': 1}],
                            'params': []}
    TTL_index += 1

    # ----------------------------------
    # Tone sequence
    # ----------------------------------
    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-03-51_ToneSequence_StimGen.pickle'), 'rb') as f:
        stimgen = pickle.load(f)

    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-03-51_ToneSequence_ExtraData.pickle'), 'rb') as f:
        extra_data = pickle.load(f)

    params = {'frequencies': extra_data['frequencies'],
              'levels': extra_data['levels'],
              'samplerate': extra_data['samplerate']}
    events = extra_data['events']

    n_events = len(events)
    ts = {'Angled': [TTL_probes['Angled'][TTL_index:TTL_index+n_events]],
          'Vertical': [TTL_probes['Vertical'][TTL_index:TTL_index+n_events]]}

    stim_data['ToneSequence'] = {'duration': stimgen.get_duration(),
                                 'timestamps': ts,
                                 'samplerate': samplerate,
                                 'events': events,
                                 'params': params}
    TTL_index += len(events)

    # ----------------------------------
    # noise bursts
    # ----------------------------------
    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-10-32_NoiseBurst_StimGen.pickle'), 'rb') as f:
        stimgen = pickle.load(f)

    n_trials = stimgen.n_trials
    params = {'level': stimgen.level,
              'ITOI': stimgen.ITI,
              'burst_len': stimgen.burst_len,
              'samplerate': stimgen.samplerate,
              'n_trials': n_trials}

    events = [{'Trial': i+1} for i in range(n_trials)]

    ts = {'Angled': [TTL_probes['Angled'][TTL_index:TTL_index+n_trials]],
          'Vertical': [TTL_probes['Vertical'][TTL_index:TTL_index+n_trials]]}

    stim_data['NoiseBursts'] = {'duration': stimgen.get_duration(),
                                'timestamps': ts,
                                'samplerate': samplerate,
                                'events': events,
                                'params': params}
    TTL_index += n_trials

    # ----------------------------------
    # FM sweeps
    # ----------------------------------
    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-12-14_FMSweep_StimGen.pickle'), 'rb') as f:
        stimgen = pickle.load(f)

    n_trials = stimgen.n_trials
    params = {'level': stimgen.level,
              'ITOI': stimgen.ITI,
              'sweep_len': stimgen.sweep_len,
              'f_start': stimgen.f_start,
              'f_stop': stimgen.f_stop,
              'samplerate': stimgen.samplerate,
              'n_trials': n_trials}

    events = [{'Trial': i+1} for i in range(n_trials)]
    ts = {'Angled': [TTL_probes['Angled'][TTL_index:TTL_index+n_trials]],
          'Vertical': [TTL_probes['Vertical'][TTL_index:TTL_index+n_trials]]}

    stim_data['FMSweeps'] = {'duration': stimgen.get_duration(),
                             'timestamps': ts,
                             'samplerate': samplerate,
                             'events': events,
                             'params': params}
    TTL_index += n_trials

    # ----------------------------------
    # DRC
    # ----------------------------------
    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-13-55_DRC_StimGen.pickle'), 'rb') as f:
        stimgen = pickle.load(f)

    with open(op.join(stimgen_path, 'recording_03_rat_acute_probe_basic',
                      '2017-03-28_19-13-55_DRC_ExtraData.pickle'), 'rb') as f:
        extra_data = pickle.load(f)

    n_trials = stimgen.n_trials
    params = {'level_lower': stimgen.level_lower,
              'level_upper': stimgen.level_upper,
              'level_step': stimgen.level_step,
              'ITI': stimgen.ITI,
              'chord_len': stimgen.chord_len,
              'density': stimgen.density,
              'trial_duration': stimgen.duration,
              'f_center': stimgen.get_center_frequencies(),
              'samplerate': stimgen.samplerate,
              'n_trials': n_trials,
              'mask': extra_data['mask']}

    events = [{'Trial': i+1} for i in range(n_trials)]
    ts = {'Angled': [TTL_probes['Angled'][TTL_index:TTL_index+n_trials]],
          'Vertical': [TTL_probes['Vertical'][TTL_index:TTL_index+n_trials]]}

    stim_data['DRC'] = {'duration': stimgen.get_duration(),
                        'timestamps': ts,
                        'samplerate': samplerate,
                        'events': events,
                        'params': params}
    TTL_index += n_trials

    # ----------------------------------
    # wave files
    # ----------------------------------
    stimgen_files = glob.glob(op.join(stimgen_path,
                                      'recording_04_rat_acute_probe_music',
                                      '*.pickle'))

    for sf in sorted(stimgen_files)[:3]:

        with open(sf, 'r') as f:
            stimgen = pickle.load(f)

        wav_file = op.join(wav_path, op.split(stimgen.filepath)[1])
        fs, samples = sio.wavfile.read(wav_file)
        stim_len = samples.shape[0] / float(fs)

        n_trials = stimgen.n_trials
        params = {'samples': samples,
                  'samplerate': samplerate,
                  'stim_len': stim_len,
                  'file': op.split(stimgen.filepath)[1]}

        events = [{'Trial': i+1} for i in range(n_trials)]
        ts = {'Angled': [TTL_probes['Angled'][TTL_index:TTL_index+n_trials]],
              'Vertical': [TTL_probes['Vertical'][TTL_index:TTL_index+n_trials]]}

        # convert file name to camel case
        name = op.splitext(op.split(stimgen.filepath)[1])[0][:-21]
        name = ''.join([x[0].upper() + x[1:] for x in name.split('_')])
        stim_data[name] = {'duration': n_trials * stim_len,
                           'timestamps': ts,
                           'samplerate': samplerate,
                           'events': events,
                           'params': params}

        TTL_index += n_trials

    print(TTL_probes['Angled'].shape, TTL_probes['Vertical'].shape, TTL_index)

    file_path = op.join(stim_path, 'stim_data')
    np.save(file_path + '.npy', stim_data)
    with open(file_path + '.pickle', 'w') as f:
        pickle.dump(stim_data, f)

    # TODO: plot color-coded timestamps for all stimuli (using vlines)


if __name__ == '__main__':
    convert_stim(sys.argv[1])