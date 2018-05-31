#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    basic analysis (signal power, tuning curves etc) of double probe data

    test /home/arne/research/data/Neuroseeker_2017_03_28_Auditory_DoubleProbes

"""

from __future__ import print_function

import click
import os.path as op
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import visvis as vv
import pickle
import operator
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import pandas as pd

TURN_CLI_ON = False


# new matplotlib color scheme
# https://matplotlib.org/users/dflt_style_changes.html
NEW_MPL_COLORS = ['#1f77b4',  # blue
                  '#ff7f0e',  # orange
                  '#2ca02c',  # green
                  '#d62728',  # red
                  '#9467bd',  # purple
                  '#8c564b',  # brown
                  '#e377c2',  # pink
                  '#7f7f7f',  # gray
                  '#bcbd22',  # ocher
                  '#17becf'  # turquois
                  ]


# probe/region information
# TODO: make this part of ProbeData class
position_mult = 2.25
probe_dimensions = [100, 8100]

brain_regions_v = {'AuD': 5208,
                   'Au1': 4748,
                   'AuV': 2890,
                   'TeA': 2248,
                   'Ectorhinal': 1933,
                   'Perirhinal': 1418,
                   'Entorhinal': 808}

brain_regions_a = {'Au1 L1-3': 6396,
                   'Au1 L4': 5983,
                   'Au1 L5': 5783,
                   'Au1 L6': 5443,
                   'CA1': 5038,
                   'Dentate \nGyrus': 4353,
                   'MGN': 2733,
                   'Substantia \nNigra': 1440}

brain_regions = {'Angled': brain_regions_a,
                 'Vertical': brain_regions_v}



def crosscorrelate_spike_trains(spike_times_train_1, spike_times_train_2, lag=None):
    if spike_times_train_1.size < spike_times_train_2.size:
        if lag is None:
            lag = np.ceil(10 * np.mean(np.diff(spike_times_train_1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20 * np.mean(np.diff(spike_times_train_2)))
        spike_times_train_1, spike_times_train_2 = spike_times_train_2, spike_times_train_1
        reverse = True

    # calculate cross differences in spike times
    differences = np.array([])
    for k in np.arange(0, spike_times_train_1.size):
        differences = np.append(differences, spike_times_train_1[k] - spike_times_train_2[np.nonzero(
            (spike_times_train_2 > spike_times_train_1[k] - lag)
            & (spike_times_train_2 < spike_times_train_1[k] + lag)
            & (spike_times_train_2 != spike_times_train_1[k]))])
    if reverse is True:
        differences = -differences
    norm = np.sqrt(spike_times_train_1.size * spike_times_train_2.size)
    return differences, norm


def corrlag(x, y, maxlag=1000, normalize=True):
    """correlation function with max. time lag"""

    assert x.shape[0] == y.shape[0]

    N = x.shape[0]
    i1 = N-1 - maxlag
    i2 = N-1 + maxlag+1

    cxy = signal.correlate(x - x.mean(), y - y.mean(), 'full')

    if normalize:
        cc = np.diag(np.corrcoef(x, y), 1)
        cxy = cxy / cxy[N-1] * cc

    return cxy[i1:i2], np.arange(-maxlag, maxlag+1)


class ProbeData(object):

    def __init__(self, name, spike_times, spike_templates, marking,
                 samplerate=20000., path=None):

        self.name = name
        self.spike_times = spike_times
        self.spike_templates = spike_templates
        self.template_marking = marking
        self.samplerate = samplerate
        self.path = path

        # valid units (=templates) in recording
        n_templates = len(self.template_marking)
        good = np.arange(n_templates)[self.template_marking > 0]
        found = np.unique(self.spike_templates)
        self.good_units = np.intersect1d(good, found)

    @staticmethod
    def from_path(name, path):

        cls = np.load(op.join(path, 'spike_clusters.npy'))
        ts = np.load(op.join(path, 'spike_times.npy'))
        marking = np.load(op.join(path, 'template_marking.npy'))

        return ProbeData(name, ts, cls, marking, path=path)

    def get_spikes(self, unit, t_start=0, t_stop=np.Inf):

        ts = self.spike_times[self.spike_templates == unit] / self.samplerate
        ts = ts[np.logical_and(ts >= t_start, ts <= t_stop)]

        return ts


def load_data_raw(data_path):

    # load probe data
    angled_path = op.join(data_path, 'Angled', 'Analysis',
                          'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                          'Kilosort')
    vert_path = op.join(data_path, 'Vertical', 'Analysis',
                        'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                        'Kilosort')

    probes = {'Vertical': ProbeData.from_path('Vertical', vert_path),
              'Angled': ProbeData.from_path('Angled', angled_path)}

    spiketrains = {}
    stim = {}
    for name, probe in probes.iteritems():

        spiketrains[name] = []
        for i, u in enumerate(probe.good_units):

            ts = probe.get_spikes(u)
            spiketrains[name].append(ts)

        # get stimulus intervals
        stim_data = np.load(op.join(data_path, 'Stimuli',
                                    'stim_data.npy')).item()
        stim[name] = {}
        for stim_name, dd in stim_data.iteritems():

            T = dd['duration']
            fs = float(dd['samplerate'])
            stim_ts = np.asarray(dd['timestamps'][name]).ravel() / fs

            t0 = stim_ts[0]
            stim[name][stim_name] = {'duration': dd['duration'],
                                     'times': stim_ts,
                                     't_start': t0,
                                     't_stop': t0 + T,
                                     'events': dd['events'],
                                     'params': dd['params']}

            print("{}, t0={:.2f}, duration={:.2f}, # events={}".format(
                stim_name, t0, T, len(dd['events'])))

    return spiketrains, stim


def load_data_neo(data_path):

    stim_data = np.load(op.join(data_path, 'Stimuli', 'stim_data.npy'),encoding='latin1').item()

    # load probe data
    angled_path = op.join(data_path, 'Angled', 'Analysis',
                          'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                          'Kilosort')
    vert_path = op.join(data_path, 'Vertical', 'Analysis',
                        'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                        'Kilosort')

    probes = {'Vertical': ProbeData.from_path('Vertical', vert_path),
              'Angled': ProbeData.from_path('Angled', angled_path)}

    # store all data into a neo block object
    import neo

    blocks = {}
#    block = neo.Block()
    for name, probe in probes.items():

        print("loading data for probe:", name)

        block = neo.Block(name=name)
        rcg = neo.RecordingChannelGroup(name=name)

        for u in probe.good_units:
            unit = neo.Unit(name=name + str(u),
                            template=u,
                            probe=name)
            rcg.units.append(unit)

        block.recordingchannelgroups.append(rcg)

        for stim_name, dd in stim_data.items():

            print("  stimulus:", stim_name)

            T = dd['duration']
            fs = float(dd['samplerate'])
            params = dd['params']
            events = dd['events']

            seg = neo.Segment(name=stim_name, params=params, events=events,
                              probe=name)

#            for name, path in [('Angled', angled_path),
#                               ('Vertical', vert_path)]:

            stim_ts = np.asarray(dd['timestamps'][name]).ravel()

            # spiketrains
#                probe = probes[name]
            t0 = stim_ts[0] / fs

#                rcg = [r for r in block.recordingchannelgroups
#                       if r.name == name][0]

            for i, u in enumerate(probe.good_units):

                ts = probe.get_spikes(u, t_start=t0, t_stop=t0+T)
                train = neo.SpikeTrain(ts - t0, T,
                                       sampling_rate=fs,
                                       units='s',
                                       probe=name,
                                       template=u,
                                       index=i)
                train.t_start = 0

                train.unit = rcg.units[i]
                rcg.units[i].spiketrains.append(train)
                seg.spiketrains.append(train)

            # merge event network messages and TTL timestamps to get
            # precise information about "what" and "when"
            for j, aa in enumerate(events):
                ev = neo.Event(stim_ts[j]/fs - t0,
                               '{}{}'.format(name, j+1),
                               event_data=aa, probe=name)
                seg.events.append(ev)

            block.segments.append(seg)

        blocks[name] = block

    return blocks


def load_template_positions(data_path):

    pos_a = np.load(op.join(data_path,
                            'weighted_template_positions_angled.npy'))

    pos_v = np.load(op.join(data_path,
                            'weighted_template_positions_vertical.npy'))

    return {'Angled': pos_a * position_mult,
            'Vertical': pos_v * position_mult}


def load_avg_templates(data_path, probe, template_ind):

    angled_path = op.join(data_path, 'Angled', 'Analysis',
                          'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                          'Kilosort')
    vert_path = op.join(data_path, 'Vertical', 'Analysis',
                        'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07',
                        'Kilosort')

    if probe == 'Vertical':
        path = angled_path
    elif probe == 'Angled':
        path = vert_path

    templates = np.load(op.join(path, 'avg_spike_template.npy'))
    channel_pos = np.load(op.join(path, 'channel_positions.npy'))

    return templates[template_ind, :, :], channel_pos


def get_brain_region(regions, y_pos):

    sorted_keys = sorted(regions.items(), key=operator.itemgetter(1))

    y0 = 0
    region_pos = None
    for i, (region, depth) in enumerate(sorted_keys):

        y1 = regions[region]

        if np.logical_and(y_pos >= y0, y_pos < y1):
            region_pos = region
            break

        y0 = y1

    return region_pos


def set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8):

    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both', which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname('Arial')

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname('Arial')

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname('Arial')

    if size_text is not None:
        for at in ax.texts:
            at.set_fontsize(size_text + add_size)
            at.set_fontname('Arial')


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def view_spike_positions(spike_positions, brain_regions, probe_dimensions,
                         colors=None, x_shift=-40, y_shift=0, font_size=12,
                         ax=None):
    """
    Plot spike positions as a scatter plot on a probe marked with brain regions

    Parameters
    ----------
    spike_positions: (np.array((N,2))) the x,y positions of the spikes
    brain_regions: (dict) a dictionary with keys the names of the brain regions
                          underneath the demarcating lines and
    values the y position on the probe of the demarcating lines
    probe_dimensions: (np.array(2)) the x and y limits of the probe
    Returns
    -------
    """

    if ax is None:
        fig, ax = plt.subplots()

    if colors is None:
        colors = NEW_MPL_COLORS[0]

    ax.scatter(spike_positions[:, 0], spike_positions[:, 1], s=5,
               color=colors)

    sorted_keys = sorted(brain_regions.items(), key=operator.itemgetter(1))

    y0 = 0
    for i, (region, depth) in enumerate(sorted_keys):

        y1 = brain_regions[region]
        ax.text(x_shift, y0 + .5*(y1 - y0) - y_shift, region,
                va='center', ha='left', fontsize=font_size)

        ax.axhline(y1, ls='--', color=3*[.25], linewidth=1)

        y0 = y1

    ax.axvspan(0, probe_dimensions[0], color=3*[.75], zorder=-1)

    ax.set_xlabel(r'Distance from edge ($\mu\mathrm{m})$')
    ax.set_ylabel(r'Distance from tip ($\mu\mathrm{m})$')
    ax.set_xlim(-50, probe_dimensions[0]+10)
    ax.set_ylim(0, y0+100)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.tick_params(axis='y', direction='in', length=5, width=1,
                   colors='k')

    return ax


@click.command(name='raw')
@click.argument('data_path', type=click.Path(exists=True))
def cli_raw(data_path=None):

    probes, stim_data = load_data_raw(data_path)

    app = vv.use()
    for probe_name, spiketrains in probes.items():

        t_min = min([t.min() for t in spiketrains])
        t_max = max([t.max() for t in spiketrains])
        n_units = len(spiketrains)

        fig = vv.figure()
        ax = vv.subplot(111)

        ax.axis.visible = 1
        ax.axis.xLabel = 'Time (s)'
        ax.axis.yLabel = 'Unit'
        ax.daspectAuto = True
        ax.SetLimits(rangeX=(t_min, t_max),
                     rangeY=(0, n_units))
        ax.cameraType = '2d'

        for i, train in enumerate(spiketrains):

            vv.plot(train, i*np.ones_like(train),
                    ls=None, mw=2, mc='k', ms='d',
                    mew=0, mec='k', alpha=1, axesAdjust=False, axes=ax)

        stim = stim_data[probe_name]
        for j, (stim_name, dd) in enumerate(stim.items()):

            vv.plot(2*[dd['t_start']], [0, n_units], lw=3, lc=3*[.5], ls="-",
                    alpha=1, axesAdjust=False, axes=ax)

#            for ts in dd['times']:
#                vv.plot(2*[ts], [0, n_units], lw=1, lc=3*[.5], ls="-",
#                        alpha=.5, axesAdjust=False, axes=ax)

        fig.DrawNow()

    app.ProcessEvents()
#    vv.screenshot(op.join(op.expanduser('~'), 'Desktop',
#                          'DoubleProbeACx_{}.png'.format(stimulus)),
#                  sf=3, bg='w')
    app.Run()


@click.command(name='raster')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--stimulus', '-s', default='NoiseBursts')
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--mgn-only', '-m', is_flag=True)
def cli_raster(data_path=None, stimulus=None, overwrite=False, mgn_only=False):

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            block = pickle.load(f)
    else:
        block = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'wb') as f:
            pickle.dump(block, f)

    # get segment for given stimulus
    seg = [s for s in block.segments if s.name == stimulus][0]
    spiketrains = seg.spiketrains

    if mgn_only:
        # show only MGN neurons
        template_positions = load_template_positions(op.expanduser(data_path))
        pos = template_positions['Angled']

        y2 = brain_regions_a['MGN']
        y1 = brain_regions_a['Substantia \nNigra']
        valid = np.logical_and(pos[:, 1] >= y1,
                               pos[:, 1] <= y2)
        spiketrains = [train for v, train in zip(valid, seg.spiketrains)
                       if v == 1]

    n_units = len(spiketrains)

    # plot spike rasters
    app = vv.use()
    fig = vv.figure()
    ax = vv.subplot(111)

    ax.axis.visible = 1
    ax.axis.xLabel = 'Time (s)'
    ax.axis.yLabel = 'Unit'
    ax.daspectAuto = True
    ax.SetLimits(rangeX=(seg.t_start, seg.t_stop),
                 rangeY=(0, n_units))
    ax.cameraType = '2d'

    for i, train in enumerate(spiketrains):

        print(i+1, n_units)
        vv.plot(train.magnitude, i*np.ones_like(train.magnitude),
                ls=None, mw=2, mc='k', ms='d',
                mew=0, mec='k', alpha=1, axesAdjust=False, axes=ax)

#    for ev in seg.events:
#        if ev.annotations['trigger'] == 'Angled':
#            vv.plot(2*[ev.time], [0, n_units], lc=(.5, .5, .5), alpha=.5,
#                    axes=ax, axesAdjust=False)

    fig.DrawNow()
    app.ProcessEvents()
    vv.screenshot(op.join(op.expanduser('~'), 'Desktop',
                          'DoubleProbeACx_{}.png'.format(stimulus)),
                  sf=3, bg='w')
    app.Run()


@click.command(name='positions')
@click.argument('data_path', type=click.Path(exists=True))
def cli_positions(data_path=None):

    template_positions = load_template_positions(op.expanduser(data_path))

    fig = plt.figure()

    for i, (name, regions) in enumerate([('Angled', brain_regions_a),
                                         ('Vertical', brain_regions_v)]):

        ax = fig.add_subplot(1, 2, i+1)
        pos_cor = template_positions[name]
        view_spike_positions(pos_cor,
                             brain_regions=regions,
                             probe_dimensions=probe_dimensions,
                             ax=ax)
        ax.set_title(name)
        simple_xy_axes(ax)
        set_font_axes(ax, add_size=5, size_labels=12, size_title=11)

    fig.set_size_inches(7.2, 8)
    fig.tight_layout(pad=.85, h_pad=.25, w_pad=.85)

    for ff in ['pdf', 'png']:
        fig.savefig(op.join(op.expanduser(data_path), 'Analysis',
                            'templates_probe.' + ff), dpi=600)
    plt.show()

'''
@click.command(name='power')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--binwidth', '-b', default=.1)
@click.option('--stimulus', '-s', default='DRC')
@click.option('--spike-count', '-S', is_flag=True)
def cli_srfpower(data_path=None, overwrite=False, binwidth=None,
                 stimulus=None, spike_count=False):

    from lnpy import metrics

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'r') as f:
            block = pickle.load(f)
    else:
        block = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'w') as f:
            pickle.dump(block, f)

    # compute signal and noise power for each unit
    seg = [s for s in block.segments if s.name == stimulus][0]
    params = seg.annotations['params']

    print("Stimulus:", stimulus)
    if stimulus == 'FMSweeps':
        T = params['ITOI']
    elif stimulus == 'DRC':
        T = params['trial_duration']

    n_units = len(seg.spiketrains)
    P_signal = np.zeros((n_units,))
    P_noise = np.zeros((n_units,))
    E_signal = np.zeros((n_units,))
    num_spikes = np.zeros((n_units,))
    edges = np.arange(0, round(T/binwidth+.5)) * binwidth

    probe_ind = np.zeros((n_units,))
    probe_names = ['Angled', 'Vertical']

    for i, train in enumerate(seg.spiketrains[:n_units]):

        print(i+1, len(seg.spiketrains), len(train))

        trials = []
        for ev in seg.events:
            if ev.annotations['trigger'] == train.annotations['trigger']:
                cnt, _ = np.histogram(train.magnitude - ev.time, bins=edges,
                                      range=(0, T))
                trials.append(cnt)
        trials = np.asarray(trials)

        ps, pn, errs = metrics.srfpower(trials)
        P_signal[i] = ps
        P_noise[i] = pn
        E_signal[i] = errs
        num_spikes[i] = np.sum(trials)

        probe_ind[i] = probe_names.index(train.annotations['trigger'])

    # plot on probe
    template_positions = load_template_positions(op.expanduser(data_path))

    for what in ['signal_power', 'spike_count']:

        fig = plt.figure()

        for i, (probe, regions) in enumerate([('Angled', brain_regions_a),
                                             ('Vertical', brain_regions_v)]):

            ax = fig.add_subplot(1, 2, i+1)
            pos_cor = template_positions[probe]
            v = probe_ind == probe_names.index(probe)

            if what == 'signal_power':
                # signal power
                y = P_signal[v]
                y /= y.max()
                y[y <= 1e-6] = 1e-6
                y = np.log(y)
                vmin = np.log(1e-6)
                vmax = 0
            else:
                # spike count
                y = num_spikes[v]
                vmin = 0
                vmax = y.max()

            cm = plt.cm.hot_r
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=cm)
            colors = [cmap.to_rgba(yi) for yi in y]

            view_spike_positions(pos_cor,
                                 brain_regions=regions,
                                 probe_dimensions=probe_dimensions,
                                 colors=colors,
                                 ax=ax)
            ax.set_title(probe)
            simple_xy_axes(ax)
            set_font_axes(ax, add_size=5, size_labels=12, size_title=11)

        fig.set_size_inches(7.2, 8)
        fig.tight_layout(pad=.85, h_pad=.25, w_pad=.85)

        fig_file = op.join(op.expanduser(data_path),
                           'Analysis',
                           what + '.')
        for ff in ['pdf', 'png']:
            fig.savefig(fig_file + ff, format=ff, dpi=600)

    plt.show()


@click.command(name='srf')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--maxlag', '-l', default=0.2)
def cli_srfs(data_path=None, overwrite=False, maxlag=None):

    data_path = op.expanduser(data_path)

    temp_file = op.join(data_path, 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'r') as f:
            block = pickle.load(f)
    else:
        block = load_data_neo(data_path)
        with open(temp_file, 'w') as f:
            pickle.dump(block, f)

    # compute signal and noise power for each unit
    seg = [s for s in block.segments if s.name == 'DRC'][0]
    params = seg.annotations['params']
    T = params['trial_duration']
    mask = params['mask']
    binwidth = params['chord_len']

    n_units = len(seg.spiketrains)
    edges = np.arange(0, round(T/binwidth+.5)) * binwidth

    # the stimulus is the same for all cells
    from lnpy.util import segment, makedirs_save
    from lnpy.linear import Ridge

    n_frequencies = mask.shape[1]
    shift = n_frequencies
    lag = int(round(maxlag / binwidth + .5))
    seg_len = lag * n_frequencies
    rfsize = (lag, n_frequencies)
    XX = segment(mask.ravel(), seg_len, shift, zero_padding=True)

    # output path
    fig_path = op.join(data_path, 'Analysis', 'SRFs')
    makedirs_save(fig_path)

    # to get information about brain area
    template_pos = load_template_positions(data_path)

    for i, train in enumerate(seg.spiketrains[:n_units]):

        trigger = train.annotations['trigger']
        print(i+1, len(seg.spiketrains), len(train), trigger)

        trials = []
        for ev in seg.events:
            if ev.annotations['trigger'] == trigger:
                cnt, _ = np.histogram(train.magnitude - ev.time, bins=edges,
                                      range=(0, T))
                trials.append(cnt)
        trials = np.asarray(trials)

        # position on probe
        pos_cor = template_pos[trigger]
        db()

        y = np.mean(trials, axis=0)
        yy = y[lag-1:]

        try:
            estimator = Ridge()
            estimator.fit(XX, yy)
            fig = estimator.show(shape=rfsize, dt=binwidth, cmap='RdBu_r',
                                 show_now=False)

            fig_file = op.join(fig_path, 'unit_{}_Ridge.png'.format(i))
            fig.savefig(fig_file, format='png', dpi=300)
            plt.close(fig)

        except:
            traceback.print_exc()
'''

@click.command(name='pca')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--binwidth', '-b', default=.1)
@click.option('--stimulus', '-s', default='Silence')
def cli_pca(data_path=None, overwrite=False, binwidth=None,
            stimulus=None):

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            block = pickle.load(f)
    else:
        block = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'wb') as f:
            pickle.dump(block, f)

    # compute signal and noise power for each unit
    seg = [s for s in block.segments if s.name == stimulus][0]

    print("Stimulus:", stimulus)

    n_units = len(seg.spiketrains)
    trials = []
    T = seg.t_stop.item() - seg.t_start.item()
    bins = seg.t_start.item() + np.arange(int(round(T/binwidth+.5))) * binwidth
    min_rate = 0

    for i, train in enumerate(seg.spiketrains[:n_units]):

        print(i+1, len(seg.spiketrains))

        cnt, _ = np.histogram(train.magnitude, bins=bins)
        if np.sum(cnt) / T >= min_rate:
            trials.append(cnt)

    trials = np.asarray(trials)

    print("computing PCA")
    from sklearn.decomposition import PCA
    pca = PCA(whiten=True)
    pca.fit(trials.T)

    fig, axarr = plt.subplots(nrows=1, ncols=1)

    ax = axarr
    ax.plot(np.cumsum(pca.explained_variance_ratio_)*100, '-',
            color=NEW_MPL_COLORS[0])
    ax.set_xlabel('# components')
    ax.set_ylabel('Explained variance (%)')
    ax.set_xlim(-5, n_units+5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.axhline(90, ls='--', color=3*[.5])

    plt.show()


@click.command(name='tuning')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--binwidth', '-b', default=.05)
@click.option('--stimulus', '-s', default='FMSweeps')
@click.option('--plot-type', '-p', default='psth')
@click.option('--pre', '-p', default=0.5)
@click.option('--post', '-P', default=0.5)
@click.option('--test-pre', '-t', default=0.5)
@click.option('--test-post', '-T', default=0.1)
@click.option('--min-rate', '-r', default=1.)
def cli_tuning(data_path=None, overwrite=False, binwidth=None,
               stimulus=None, pre=None, post=None, plot_type=None,
               test_pre=None, test_post=None, min_rate=None):
    tuning(data_path=data_path, overwrite=overwrite, binwidth=binwidth,
           stimulus=stimulus, pre=pre, post=post, plot_type=plot_type,
           test_pre=test_pre, test_post=test_post, min_rate=min_rate)


def tuning(data_path=None, overwrite=True, binwidth=0.05,
           stimulus='FMSweeps', frequencies_index=None, pre=0.5, post=0.5, plot_type='psth',
           test_pre=0.5, test_post=0.1, min_rate=1.):

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            blocks = pickle.load(f)
    else:
        blocks = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'wb') as f:
            pickle.dump(blocks, f)

    template_positions = load_template_positions(op.expanduser(data_path))

    pdf_file = op.join(op.expanduser(data_path), 'Analysis',
                       'tuning_{}_{}.pdf'.format(stimulus, plot_type))
    cols_per_page = 5
    rows_per_page = 10

    print("Stimulus:", stimulus)

    with PdfPages(pdf_file) as pdf:

        for probe_name, block in blocks.items():

            print("  processing probe:", probe_name)

            seg = [s for s in block.segments if s.name == stimulus][0]
            params = seg.annotations['params']

            if stimulus == 'FMSweeps':
                T = params['sweep_len']
            elif stimulus == 'DRC':
                T = params['trial_duration']
            elif stimulus == 'NoiseBursts':
                T = params['burst_len']
            elif stimulus == 'ToneSequence':
                T = np.unique([ev.annotations['event_data']['Duration']
                               for ev in seg.events])

            T_total = float(seg.t_stop) - float(seg.t_start)

            n_pre = int(round(pre / binwidth + .5))
            n_post = int(round(post / binwidth + .5))
            edges = np.arange(-n_pre, np.round(T/binwidth+.5)+n_post) * binwidth
            regions = brain_regions[probe_name]
            pos_cor = template_positions[probe_name]
            sorted_keys = sorted(regions.items(), key=operator.itemgetter(1))

            previous_depth = 0
            for j, (region, depth) in enumerate(sorted_keys):

                print("   region:", region)

                # cells for this brain region
                v = np.logical_and(pos_cor[:, 1] >= previous_depth,
                                   pos_cor[:, 1] < depth)
                ind = np.where(v)[0]

                # only analyze cells with minimum numbers of spikes
                rate_cells = np.asarray([len(seg.spiketrains[u]) / T_total
                                         for u in ind])
                print("      # cells:", rate_cells.shape[0])
                ind = ind[rate_cells >= min_rate]
                rate_cells = rate_cells[rate_cells >= min_rate]
                print("      # cells with rate > {:.2f} Hz:".format(min_rate),
                      rate_cells.shape[0])

                fig = plt.figure(figsize=(7.1, 12))
                fig.suptitle(region)

                ax = None
                plot_num = 0
                n_sig = 0

                for ii, index in enumerate(ind):

                    train = [t for t in seg.spiketrains
                             if t.annotations['index'] == index][0]
                    ts = train.magnitude
                    template = train.annotations['template']

                    plot_num += 1
                    ax = fig.add_subplot(rows_per_page, cols_per_page,
                                         plot_num, sharex=ax)

                    if plot_type in ['psth', 'PSTH']:

                        trials = []
                        for ev in seg.events:

                            if stimulus != 'ToneSequence':
                                cnt, _ = np.histogram(ts - ev.time,
                                                      bins=edges)
                                trials.append(cnt)
                            elif ev.annotations['event_data']['Level'] <= 70:
                                if frequencies_index is not None:
                                    fs = seg.annotations['params']['frequencies'][frequencies_index]
                                    if ev.annotations['event_data']['Frequency'] in fs:
                                        cnt, _ = np.histogram(ts - ev.time,
                                                              bins=edges)
                                        trials.append(cnt)
                                else:
                                    cnt, _ = np.histogram(ts - ev.time,
                                                          bins=edges)
                                    trials.append(cnt)

                        trials = np.asarray(trials)

                        ax.bar(edges[:-1]*1000,
                               np.mean(trials, axis=0) / binwidth,
                               color=NEW_MPL_COLORS[0],
                               width=binwidth*1000,
                               ec='none')
                        ax.set_ylabel('Rate (Hz)')

                        ax.axvspan(0, T * 1000, color=3 * [.5], alpha=.5,
                                   lw=0)
                        ax.set_xlabel('Time (ms)')
                        ax.set_xlim(-pre * 1000, (T + post) * 1000)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                    elif plot_type in ['raster', 'Raster']:
                        y0 = 1
                        for ev in seg.events:
                            t0 = ev.time
                            v = np.logical_and(ts - t0 >= -pre,
                                               ts - t0 < T + post)

                            ax.plot((train[v]-t0)*1000,
                                    y0*np.ones((v.sum(),)),
                                    'd', ms=1, color=3*[.25],
                                    mec='none')
                            y0 += 1

                        ax.axvspan(0, T * 1000, color=3 * [.5], alpha=.5,
                                   lw=0)
                        ax.set_xlabel('Time (ms)')
                        ax.set_xlim(-pre * 1000, (T + post) * 1000)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                    elif plot_type in ['autocorrelogram', 'Autocorrelogram']:
                        T = ts.max()
                        maxlag = pre
                        N = int(round(T / binwidth + .5))
                        bins = np.arange(N) * binwidth
                        cnt = np.histogram(ts, bins=bins)[0]
                        cc, lags = corrlag(cnt, cnt, maxlag=int(round(maxlag / binwidth + .5)))

                        ax.plot(lags * binwidth, cc, '-', lw=0.5, color='#1f77b4')
                        ax.set_xlabel('Time (ms)')
                        ax.set_ylabel('Correlation')

#                    ax.axvline(-test_pre * 1000, ls='--', lw=.5, color=3*[.5])
#                    ax.axvline((T + test_post) * 1000, ls='--',
#                               lw=.5, color=3*[.5])

                    # test for statistical significance
                    n_events = len(seg.events)
                    spikes_pre_post = np.zeros((n_events, 2))
                    for jj, ev in enumerate(seg.events):
                        tt = ts - ev.time
                        n_pre = np.sum(np.logical_and(tt >= -test_pre,
                                                      tt < 0))
                        n_post = np.sum(np.logical_and(tt >= 0,
                                                       tt <= T+test_post))
                        spikes_pre_post[jj, :] = [n_pre / test_pre,
                                                  n_post / (T+test_post)]

                    h, pval = stats.wilcoxon(spikes_pre_post[:, 0],
                                             spikes_pre_post[:, 1])

                    title = 'unit {}'.format(template)
                    if pval <= 0.01:
                        title += ' *'
                        n_sig += 1

                    #print("     ", index, h, pval, rate_cells[ii])

                    ax.set_title(title)
                    simple_xy_axes(ax)
                    set_font_axes(ax)

                    if (plot_num+1) % (cols_per_page * rows_per_page) == 0:

                        fig.tight_layout(rect=(0, 0, 1, .95))
                        pdf.savefig(fig)
                        plt.close(fig)
                        fig = None

                        if ii < len(ind) - 1:
                            fig = plt.figure(figsize=(7.1, 12))
                            fig.suptitle(region)
                            plot_num = 0

                print("    # sign. cells:", n_sig, ind.shape[0])

                if fig is not None and plot_num > 0:
                    fig.tight_layout(rect=(0, 0, 1, .95))
                    pdf.savefig(fig)
                    plt.close(fig)

                previous_depth = depth


@click.command(name='tones')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--binwidth', '-b', default=.05)
@click.option('--pre', '-p', default=0.3)
@click.option('--post', '-P', default=0.2)
@click.option('--test-pre', '-t', default=0.3)
@click.option('--test-post', '-T', default=0.05)
@click.option('--min-rate', '-r', default=1.)
def cli_tones(data_path=None, overwrite=False, binwidth=None,
              stimulus=None, pre=None, post=None, plot_type=None,
              test_pre=None, test_post=None, min_rate=None):
    tones(data_path=data_path, overwrite=overwrite, binwidth=binwidth,
          pre=pre, post=post, test_pre=test_pre, test_post=test_post, min_rate=min_rate)


def tones(data_path=None, overwrite=True, binwidth=0.05, pre=0.3, post=0.2,
          test_pre=0.3, test_post=0.05, min_rate=1.):

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            blocks = pickle.load(f)
    else:
        blocks = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'wb') as f:
            pickle.dump(blocks, f)

    template_positions = load_template_positions(op.expanduser(data_path))

    pdf_file = op.join(op.expanduser(data_path), 'Analysis', 'tones.pdf')

    figsize = (15, 12)
    layout_opts = dict(rect=(0, 0, 1, .95),
                       pad=.1,
                       h_pad=.1,
                       w_pad=.1)

    saved_templates = []
    saved_time_diff_df = pd.DataFrame()

    with PdfPages(pdf_file) as pdf:

        for probe_name, block in blocks.items():

            print("processing probe:", probe_name)

            seg = [s for s in block.segments if s.name == 'ToneSequence'][0]
            params = seg.annotations['params']
            event_info = seg.annotations['events']

            T = np.unique([ev['Duration'] for ev in event_info])
            levels = params['levels'][:3]  # throw away the 90db
            frequencies = params['frequencies']

            T_total = float(seg.t_stop) - float(seg.t_start)

            n_pre = int(round(pre / binwidth + .5))
            n_post = int(round(post / binwidth + .5))
            edges = np.arange(-n_pre, np.round(T/binwidth+.5)+n_post) * binwidth

            n_freqs = len(frequencies)
            cols_per_page = n_freqs
            rows_per_page = 10

            regions = brain_regions[probe_name]
            pos_cor = template_positions[probe_name]
            sorted_keys = sorted(regions.items(), key=operator.itemgetter(1))

            previous_depth = 0
            for j, (region, depth) in enumerate(sorted_keys):

                print("  region:", region)

                # cells for this brain region
                v = np.logical_and(pos_cor[:, 1] >= previous_depth,
                                   pos_cor[:, 1] < depth)
                ind = np.where(v)[0]

                # only analyze cells with minimum numbers of spikes
                rate_cells = np.asarray([len(seg.spiketrains[u]) / T_total
                                         for u in ind])
                print("    # cells:", rate_cells.shape[0])
                ind = ind[rate_cells >= min_rate]
                rate_cells = rate_cells[rate_cells >= min_rate]
                print("    # cells with rate > {:.2f} Hz:".format(min_rate),
                      rate_cells.shape[0])

                fig = plt.figure(figsize=figsize)
                fig.suptitle(region)
                plot_num = 0
                n_sig = 0

                for ii, index in enumerate(ind):

                    train = [t for t in seg.spiketrains
                             if t.annotations['index'] == index][0]

                    template = train.annotations['template']

                    print(ii, len(ind), template)

                    saved_templates.append(template)

                    df = pd.DataFrame(index=[template])

                    ts = train.magnitude

                    ax = None

                    #for ll in [levels[:2],
                    #           levels[2:]]:
                    for ll in levels:

                        #print("    level:", ll)

                        for jj, f in enumerate(frequencies):

                            #print("      frequency:", f)

                            if fig is None:
                                fig = plt.figure(figsize=figsize)
                                fig.suptitle(region)
                                plot_num = 0

                            plot_num += 1
                            ax = fig.add_subplot(rows_per_page,
                                                 cols_per_page,
                                                 plot_num,
                                                 sharex=ax,
                                                 sharey=ax)
                            ax.axvspan(0, T*1000, color=3*[.5], alpha=.5,
                                       lw=0)

                            trials = []
                            for ev, info in zip(seg.events, event_info):

                                #if info['Level'] in ll and \
                                #        round(info['Frequency']) == round(f):
                                if info['Level'] == ll and \
                                                round(info['Frequency']) == round(f):
                                    cnt, _ = np.histogram(ts - ev.time,
                                                          bins=edges)
                                    trials.append(cnt)

                                    trial = info['Trial'] - 1
                                    column_name = str(ll)+'_'+str(round(f))
                                    try:
                                        temp = df[column_name].loc[template]
                                        temp[trial, :] = np.array(ts - ev.time)
                                    except:
                                        temp = np.empty((10, len(ts)))
                                        temp[trial, :] = np.array(ts - ev.time)
                                    finally:
                                        df[column_name] = pd.Series([temp], index=df.index)

                            trials = np.asarray(trials)

                            ax.bar(edges[:-1]*1000,
                                   np.mean(trials, axis=0) / binwidth,
                                   color=NEW_MPL_COLORS[0],
                                   width=binwidth*1000,
                                   ec='none')
#                            ax.set_ylabel('Rate (Hz)')

#                            ax.set_xlabel('Time (ms)')
                            ax.set_xlim(-pre*1000, (T + post)*1000)
#                            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                            ax.set_xticklabels([])

                            # test for statistical significance
                            n_events = len(seg.events)
                            spikes_pre_post = np.zeros((n_events, 2))
                            for jj, ev in enumerate(seg.events):
                                tt = ts - ev.time
                                n_pre = np.sum(np.logical_and(tt >= -test_pre,
                                                              tt < 0))
                                n_post = np.sum(np.logical_and(tt >= 0,
                                                               tt <= T+test_post))
                                spikes_pre_post[jj, :] = [n_pre / test_pre,
                                                          n_post / (T+test_post)]

                            h, pval = stats.wilcoxon(spikes_pre_post[:, 0],
                                                     spikes_pre_post[:, 1])

                            title = '{},{},{}'.format(template, str(ll), str(round(f/1000)))
                            if pval <= 0.01:
                                title += ' *'
                                n_sig += 1

                            ax.set_title(title)
                            simple_xy_axes(ax)
                            set_font_axes(ax, add_size=-2)

                            if (plot_num+1) % (cols_per_page * rows_per_page) == 0:

                                fig.tight_layout(**layout_opts)
                                pdf.savefig(fig)
                                plt.close(fig)
                                fig = None

                                if ii < len(ind):
                                    fig = plt.figure(figsize=figsize)
                                    fig.suptitle(region)
                                    plot_num = 0

                print("  # sign. cells:", n_sig, ind.shape[0])

                saved_time_diff_df = saved_time_diff_df.append(df)
                saved_time_diff_df.to_pickle(op.join(data_path, 'time_diffs_to_events.df'))

                if fig is not None and plot_num > 0:
                    fig.tight_layout(**layout_opts)
                    pdf.savefig(fig)
                    plt.close(fig)

                previous_depth = depth



@click.command(name='fra')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--binwidth', '-b', default=.05)
@click.option('--pre', '-p', default=0.0)
@click.option('--post', '-P', default=0.05)
@click.option('--min-rate', '-r', default=1.)
def cli_tones_fra(data_path=None, overwrite=False, binwidth=None,
                  pre=None, post=None, min_rate=None):

    temp_file = op.join(op.expanduser(data_path), 'data_neo_block.pickle')
    if op.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            blocks = pickle.load(f)
    else:
        blocks = load_data_neo(op.expanduser(data_path))
        with open(temp_file, 'wb') as f:
            pickle.dump(blocks, f)

    template_positions = load_template_positions(op.expanduser(data_path))

    pdf_file = op.join(op.expanduser(data_path), 'Analysis', 'tones_fra.pdf')

    cols_per_page = 10
    rows_per_page = 10

    figsize = (15, 12)
    layout_opts = dict(rect=(0, 0, 1, .95),
                       pad=.1,
                       h_pad=.1,
                       w_pad=.1)

    with PdfPages(pdf_file) as pdf:

        for probe_name, block in blocks.items():

            print("processing probe:", probe_name)

            seg = [s for s in block.segments if s.name == 'ToneSequence'][0]
            params = seg.annotations['params']
            event_info = seg.annotations['events']

            T = np.unique([ev['Duration'] for ev in event_info])
            levels = params['levels']
            frequencies = params['frequencies']
            n_trials = max([ev['Trial'] for ev in event_info])

            levels = [ll for ll in levels if ll < 90]

            T_total = float(seg.t_stop) - float(seg.t_start)

            regions = brain_regions[probe_name]
            pos_cor = template_positions[probe_name]
            sorted_keys = sorted(regions.items(), key=operator.itemgetter(1))

            previous_depth = 0
            for j, (region, depth) in enumerate(sorted_keys):

                print("  region:", region, previous_depth, depth)

                # cells for this brain region
                v = np.logical_and(pos_cor[:, 1] >= previous_depth,
                                   pos_cor[:, 1] < depth)
                ind = np.where(v)[0]
                previous_depth = depth

                # only analyze cells with minimum numbers of spikes
                rate_cells = np.asarray([len(seg.spiketrains[u]) / T_total
                                         for u in ind])
                print("    # cells:", rate_cells.shape[0])
                ind = ind[rate_cells >= min_rate]
                rate_cells = rate_cells[rate_cells >= min_rate]
                print("    # cells with rate > {:.2f} Hz:".format(min_rate),
                      rate_cells.shape[0])

                fig = plt.figure(figsize=figsize)
                fig.suptitle(region)
                plot_num = 0

                ax = None

                for ii, index in enumerate(ind):

                    train = [t for t in seg.spiketrains
                             if t.annotations['index'] == index][0]
                    ts = train.magnitude
                    template = train.annotations['template']

                    # compute FRA
                    FRA = np.zeros((len(levels), len(frequencies)))
                    PRE = np.zeros((len(levels), len(frequencies)))
                    for i1, ll in enumerate(levels):
                        for i2, ff in enumerate(frequencies):

                            for ev, info in zip(seg.events, event_info):

                                if info['Level'] == ll and \
                                        round(info['Frequency']) == round(ff):

                                    v = np.logical_and(ts - ev.time >= -pre,
                                                       ts - ev.time <= T + post)
                                    FRA[i1, i2] += np.sum(v)

                                    v = np.logical_and(ts - ev.time >= -.3,
                                                       ts - ev.time < 0)
                                    PRE[i1, i2] += np.sum(v) / .3

                    FRA = FRA / T / n_trials  # -> spike rate
                    PRE = PRE / n_trials  # -> spike rate

                    plot_num += 1
                    ax = fig.add_subplot(rows_per_page,
                                         cols_per_page,
                                         plot_num,
                                         sharex=ax,
                                         sharey=ax)

                    ax.imshow(FRA - PRE, interpolation='nearest', cmap='viridis',
                              origin='lower')
                    ax.axis('tight')
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                    title = '{}'.format(template)

                    ax.set_title(title)
                    simple_xy_axes(ax)
                    set_font_axes(ax, add_size=-2)

                    if (plot_num+1) % (cols_per_page * rows_per_page) == 0:

                        fig.tight_layout(**layout_opts)
                        pdf.savefig(fig)
                        plt.close(fig)
                        fig = None

                        if ii < len(ind):
                            fig = plt.figure(figsize=figsize)
                            fig.suptitle(region)
                            plot_num = 0

                if fig is not None and plot_num > 0:
                    fig.tight_layout(**layout_opts)
                    pdf.savefig(fig)
                    plt.close(fig)


@click.command(name='templates')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--probe', '-p', default='Angled')
@click.option('--max-index', '-i', default=50)
def cli_templates(data_path, probe=None, max_index=None):

    template_ind = range(max_index)

    templates, channel_pos = load_avg_templates(data_path,
                                                probe,
                                                template_ind)

    n_templates, n_channels, n_time = templates.shape
#    channel_pos = channel_pos * position_mult

    xc0, yc0 = np.min(channel_pos, axis=0)
    xc1, yc1 = np.max(channel_pos, axis=0)
    spike_color = NEW_MPL_COLORS[3]

    center = int(n_time / 2.)
    n_pre = 25
    n_post = 50
    i1 = center - n_pre
    i2 = center + n_post+1
    xx = np.linspace(-8, 8, n_pre+n_post+1)
    for i in range(n_templates):

        print("template", i+1, n_templates)
        ti = templates[i]
        ti_std = np.std(ti)

        fig = plt.figure(figsize=(3, 8))

        # whole probe
        ax = fig.add_axes((.05, .05, .3, .9))

        channels_zoom = []
        for j in range(n_channels):
            xc, yc = channel_pos[j]
            yy = ti[j]
            yy = yy[i1:i2]
            if yy.std() >= 2*ti_std:
                c = spike_color
                channels_zoom.append(j)
            else:
                c = 3*[.5]
            ax.plot(xc + xx, yc + yy/(3*ti_std), '-',
                    lw=.1, color=c)
        ax.axis('equal')
        ax.axis('off')

        # add scale bars
        ax.plot([0, 100], 2*[yc0 - 100], '-', color=3*[0], lw=1.5)
        ax.text(50, yc0 - 300, r'100 $\mu$m', ha='center')

        ax.plot(2*[xc0 - 100], [yc0+.1*(yc1 - yc0), yc0+.1*(yc1 - yc0)+500],
                '-', color=3*[0], lw=1.5)
        ax.text(xc0 - 250, yc0 + .1*(yc1 - yc0)+250,
                r'500 $\mu$m', ha='center', va='center', rotation=90)

        ax.set_ylim(yc0 - 350, yc1 + 200)

        # zoom-in into relevant templates
        ax = fig.add_axes((.4, .05, .55, .9))

        y_min = min([channel_pos[j, 1] for j in channels_zoom]) - 50
        y_max = max([channel_pos[j, 1] for j in channels_zoom]) + 50

        for j in range(n_channels):
            xc, yc = channel_pos[j]
            if yc >= y_min and yc <= y_max:
                yy = ti[j]
                yy = yy[i1:i2]
                if yy.std() >= 2*ti_std:
                    c = spike_color
                else:
                    c = 3*[.5]
                ax.plot(xc + xx, yc + yy/(3*ti_std), '-',
                        lw=.25, color=c)
        ax.axis('equal')
        ax.axis('off')

        # add scale bars
        ax.plot([0, 100], 2*[y_min - 10], '-', color=3*[0], lw=1.5)
        ax.text(50, y_min - 30, r'100 $\mu$m', ha='center')

        ax.plot(2*[xc0-25],
                [y_min+.1*(y_max - y_min), y_min+.1*(y_max - y_min)+100],
                '-', color=3*[0], lw=1.5)
        ax.text(xc0-40, y_min+.1*(y_max - y_min)+50,
                r'100 $\mu$m', ha='center', va='center', rotation=90)

        ax.set_ylim(y_min - 50, y_max + 50)

        for ax in fig.axes:
            set_font_axes(ax)

        fig_file = op.join(op.expanduser(data_path),
                           'Analysis',
                           'probe_templates',
                           'probe_{}_template_{}.'.format(
                               probe, template_ind[i]))
        for ff in ['pdf', 'png']:
            fig.savefig(fig_file + ff, format=ff, dpi=600)
        plt.close(fig)


@click.group()
def cli():
    pass


cli.add_command(cli_raw)
cli.add_command(cli_raster)
cli.add_command(cli_positions)
#cli.add_command(cli_srfpower)
#cli.add_command(cli_srfs)
cli.add_command(cli_pca)
#cli.add_command(cli_tuning)
cli.add_command(cli_tones)
cli.add_command(cli_tones_fra)
cli.add_command(cli_templates)


if __name__ == '__main__':
    cli()