

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def emd(x, nIMF=3, stoplim=.001):
    """Perform empirical mode decomposition to extract 'niMF' components out of the signal 'x'."""

    r = x
    t = np.arange(len(r))
    imfs = np.zeros(nIMF, dtype=object)
    for i in range(nIMF):
        r_t = r
        is_imf = False

        while is_imf == False:
            # Identify peaks and troughs
            pks = sp.signal.argrelmax(r_t)[0]
            trs = sp.signal.argrelmin(r_t)[0]

            # Interpolate extrema
            pks_r = r_t[pks]
            fip = sp.interpolate.InterpolatedUnivariateSpline(pks, pks_r, k=3)
            pks_t = fip(t)

            trs_r = r_t[trs]
            fitr = sp.interpolate.InterpolatedUnivariateSpline(trs, trs_r, k=3)
            trs_t = fitr(t)

            # Calculate mean
            mean_t = (pks_t + trs_t) / 2
            mean_t = _emd_complim(mean_t, pks, trs)

            # Assess if this is an IMF (only look in time between peaks and troughs)
            sdk = _emd_comperror(r_t, mean_t, pks, trs)

            # if not imf, update r_t and is_imf
            if sdk < stoplim:
                is_imf = True
            else:
                r_t = r_t - mean_t

        imfs[i] = r_t
        r = r - imfs[i]

    return imfs


def _emd_comperror(h, mean, pks, trs):
    """Calculate the normalized error of the current component"""
    samp_start = np.max((np.min(pks), np.min(trs)))
    samp_end = np.min((np.max(pks), np.max(trs))) + 1
    return np.sum(np.abs(mean[samp_start:samp_end] ** 2)) / np.sum(np.abs(h[samp_start:samp_end] ** 2))


def _emd_complim(mean_t, pks, trs):
    samp_start = np.max((np.min(pks), np.min(trs)))
    samp_end = np.min((np.max(pks), np.max(trs))) + 1
    mean_t[:samp_start] = mean_t[samp_start]
    mean_t[samp_end:] = mean_t[samp_end]
    return mean_t


def findnextcomponent(x,orig,t, stoplim=0.01):
    """Find the next IMF of the input signal 'x' and return it as 'r'.
    The plots are compared to the original signal, 'orig'."""
    r_t = x
    is_imf = False

    while is_imf == False:
        # Identify peaks and troughs
        pks = sp.signal.argrelmax(r_t)[0]
        trs = sp.signal.argrelmin(r_t)[0]

        # Interpolate extrema
        pks_r = r_t[pks]
        fip = sp.interpolate.InterpolatedUnivariateSpline(pks,pks_r,k=3)
        pks_t = fip(range(len(r_t)))

        trs_r = r_t[trs]
        fitr = sp.interpolate.InterpolatedUnivariateSpline(trs,trs_r,k=3)
        trs_t = fitr(range(len(r_t)))

        # Calculate mean
        mean_t = (pks_t + trs_t) / 2
        mean_t = _emd_complim(mean_t, pks, trs)

        # Assess if this is an IMF
        sdk = _emd_comperror(r_t, mean_t,pks,trs)

        # debug: make sure everything looks right, because getting weird interp error after a few iterations; error not converging?
        plt.figure(figsize=(10,1))
        plt.plot(orig,color='0.8')
        plt.plot(r_t - mean_t,'k')
        plt.ylim((-1000,1000))

        # if not imf, update r_t and is_imf
        if sdk < stoplim:
            is_imf = True
            print('IMF found!')
        else:
            r_t = r_t - mean_t

    return x - r_t