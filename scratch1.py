# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 14:43:50 2015

@author: rkmaddox
"""

import trf
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft.fftpack as fft
plt.ion()


def fft_pow2(x, n=None, axis=-1):
    if n is None:
        n = x.shape[axis]
    if n < x.shape[axis]:
        ValueError('n must not be less than the length of the data.')
    n = 2 ** int(np.ceil(np.log2(n)))
    return fft.fft(x, n, axis)


def make_xxt(ac):
    len_trf = (ac.shape[2] + 1) / 2
    n_ch = ac.shape[0]
    xxt = np.zeros([n_ch * len_trf] * 2)
    for ch0 in range(n_ch):
        for ch1 in range(n_ch):
            xxt_temp = np.zeros((len_trf, len_trf))
            xxt_temp[0, :] = ac[ch0, ch1, len_trf - 1:]
            xxt_temp[:, 0] = ac[ch0, ch1, len_trf - 1::-1]
            for i in np.arange(1, len_trf):
                xxt_temp[i, i:] = ac[ch0, ch1, len_trf - 1:-i]
                xxt_temp[i:, i] = ac[ch0, ch1, len_trf - 1:i - 1:-1]
            xxt[ch0 * len_trf:(ch0 + 1) * len_trf,
                ch1 * len_trf:(ch1 + 1) * len_trf] = xxt_temp
    return xxt


rand = np.random.RandomState(0)

# signal parameters
fs = 10e3
n_ch_in = 8
n_ch_out = 1
len_sig = fs * 1

# trf parameters
trf_start = -2e-3
trf_stop = 25e-3
trf_start_ind = int(np.floor(trf_start * fs))
trf_stop_ind = int(np.floor(trf_stop * fs))
trf_inds = np.arange(trf_start_ind, trf_stop_ind + 1, dtype=int)
t_trf = trf_inds / float(fs)
len_trf = len(t_trf)

# make the signals with some correlations
x_in = rand.randn(n_ch_in, len_sig)
x_in += rand.randn(1, len_sig)
ba = [sig.butter(6, (r0 + 0.2) / 2.4, 'lowpass') for r0 in np.arange(n_ch_in, dtype=float) / n_ch_in]

# try to figure out what is going on
x_in_filt = np.copy(x_in)
for ch in range(n_ch_in):
    x_in_filt[ch] = sig.lfilter(ba[ch][0], ba[ch][1], x_in[ch])
x_in_filt[0, :-5] = x_in_filt[0, 5:]  # advance it some samples for acausality
w_in_out = np.ones((n_ch_out, n_ch_in))  # rand.randn(n_ch_out, n_ch_in)
x_out = np.dot(w_in_out, x_in_filt) + rand.randn(n_ch_out, len_sig)

# w = (x * x.T + lam * I) \ x * y
x_in_fft = fft_pow2(x_in, x_in.shape[-1] + len_trf - 1)
x_out_fft = fft_pow2(x_out, x_out.shape[-1] + len_trf - 1)


# compute the autocorrelations
ac = np.zeros((n_ch_in, n_ch_in, len_trf * 2 - 1))
for ch0 in range(n_ch_in):
    for ch1 in np.arange(ch0, n_ch_in, dtype=int):
        ac_temp = np.real(fft.ifft(x_in_fft[ch0] * np.conj(x_in_fft[ch1])))
        ac[ch0, ch1] = np.append(ac_temp[-len_trf + 1:], ac_temp[:len_trf])
        if ch0 != ch1:
            ac[ch1, ch0] = ac[ch0, ch1][::-1]

# compute the crosscorrelations
cc = np.zeros((n_ch_out, n_ch_in, len_trf))
for ch_in in range(n_ch_in):
    for ch_out in range(n_ch_out):
        cc_temp = np.real(fft.ifft(x_out_fft[ch_out] *
                          np.conj(x_in_fft[ch_in])))
        cc[ch_out, ch_in] = np.append(cc_temp[trf_start_ind:],
                                      cc_temp[:trf_stop_ind + 1])


# make xxt and xy
xxt = make_xxt(ac) / len_sig
xy = cc.reshape([n_ch_out, n_ch_in * len_trf]) / len_sig

w = np.dot(np.linalg.pinv(xxt - 1e-5 * np.eye(n_ch_in * len_trf)), xy.T).T
w = w.reshape([n_ch_out, n_ch_in, len_trf])
plt.plot(t_trf, w[0].T)