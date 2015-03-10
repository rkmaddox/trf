# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 14:43:50 2015

@author: rkmaddox
"""

from trf import trf_corr, trf_reg
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# =============================================================================
# First make a demo input and output signal
# =============================================================================
rand = np.random.RandomState(0)

# signal parameters
fs = 200
n_ch_in = 2  # e.g., number of audio sources in
n_ch_out = 5  # e.g., number of electrodes out
len_sig = fs * 120

# trf parameters
trf_start = -100e-3
trf_stop = 300e-3

# make the signals with some correlations
x_in = rand.randn(n_ch_in, len_sig)
x_in += rand.randn(1, len_sig)
ba = [sig.butter(8, (r0 + 0.2) / 3, 'lowpass')
      for r0 in np.arange(n_ch_in, dtype=float) / n_ch_in]

# try to figure out what is going on
x_in_filt = np.copy(x_in)
for ch in range(n_ch_in):
    x_in_filt[ch] = sig.lfilter(ba[ch][0], ba[ch][1], x_in[ch])
w_in_out = rand.randn(n_ch_out, n_ch_in)
x_out = np.dot(w_in_out, x_in_filt) + 1 + rand.randn(n_ch_out, len_sig)

# =============================================================================
# Now solve for the TRF:  w = (x * x.T + lam * reg) \ x * y
# =============================================================================
# First get XX^T and XY
xxt, xy, t_trf = trf_corr(x_in, x_out, fs, trf_start, trf_stop)

# Now do inverse with some regularization
lam = 1e-1
w = trf_reg(xxt, xy, n_ch_in, lam, reg_type='laplacian')

vmax = np.abs(w).max()
for ai in range(n_ch_out):
    plt.subplot(n_ch_out, 2, 2 * ai + 1)
    plt.plot(t_trf, w[ai].T)

    plt.subplot(n_ch_out, 2, 2 * ai + 2)
    plt.imshow(w[ai], vmin=-vmax, vmax=vmax, aspect='auto',
               extent=[t_trf[0], t_trf[-1], 0, n_ch_in - 1])
