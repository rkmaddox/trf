# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 14:43:16 2015

@author: rkmaddox
"""

import numpy as np
try:  # Try to use CUDA fft
    from mne.cuda import fft, ifft
except:
    from scipy.fftpack import fft, ifft
try:  # Try to use CUDA pseudoinverse
    from skcuda import linalg
    linalg.init()
    from pycuda import gpuarray
    use_cuda_pinv = True
except:
    #from numpy.linalg import pinv
    use_cuda_pinv = False

if use_cuda_pinv:
    def pinv(a, rcond=1e-15):
        return linalg.pinv(gpuarray.to_gpu(a), rcond).get()
else:
    from numpy.linalg import pinv


def cross_correlation(x, y):
    pass


def fft_pow2(x, n=None, axis=-1):
    if n is None:
        n = x.shape[axis]
    if n < x.shape[axis]:
        ValueError('n must not be less than the length of the data.')
    n = 2 ** int(np.ceil(np.log2(n)))
    return fft(x, n, axis)


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


def trf_corr(x_in, x_out, fs, t_start, t_stop, x_in_freq=False):
    trf_start_ind = int(np.floor(t_start * fs))
    trf_stop_ind = int(np.floor(t_stop * fs))
    trf_inds = np.arange(trf_start_ind, trf_stop_ind + 1, dtype=int)
    t_trf = trf_inds / float(fs)
    len_trf = len(t_trf)
    n_ch_in, len_sig = x_in.shape
    n_ch_out = x_out.shape[0]

    if t_stop <= t_start:
        raise ValueError("t_stop must be after t_start")

    if not x_in_freq:
        x_in_fft = fft_pow2(x_in, x_in.shape[-1] + len_trf - 1)
        x_out_fft = fft_pow2(x_out, x_out.shape[-1] + len_trf - 1)
    else:
        if x_out.shape[1] > len_sig:
            raise ValueError("If x_in is in frequency domain, it must be "
                             "longer than x_out.")
        x_in_fft = x_in
        x_out_fft = fft(x_out, len_sig)

    # compute the autocorrelations
    ac = np.zeros((n_ch_in, n_ch_in, len_trf * 2 - 1))
    for ch0 in range(n_ch_in):
        for ch1 in np.arange(ch0, n_ch_in, dtype=int):
            ac_temp = np.real(ifft(x_in_fft[ch0] * np.conj(x_in_fft[ch1])))
            ac[ch0, ch1] = np.append(ac_temp[-len_trf + 1:], ac_temp[:len_trf])
            if ch0 != ch1:
                ac[ch1, ch0] = ac[ch0, ch1][::-1]

    # compute the crosscorrelations
    cc = np.zeros((n_ch_out, n_ch_in, len_trf))
    for ch_in in range(n_ch_in):
        for ch_out in range(n_ch_out):
            cc_temp = np.real(ifft(x_out_fft[ch_out] *
                              np.conj(x_in_fft[ch_in])))
            if trf_start_ind < 0 and trf_stop_ind + 1 > 0:
                cc[ch_out, ch_in] = np.append(cc_temp[trf_start_ind:],
                                              cc_temp[:trf_stop_ind + 1])
            else:
                cc[ch_out, ch_in] = cc_temp[trf_start_ind:trf_stop_ind + 1]

    # make xxt and xy
    xxt = make_xxt(ac) / len_sig
    xy = cc.reshape([n_ch_out, n_ch_in * len_trf]) / len_sig

    return xxt, xy, t_trf


def trf_reg(xxt, xy, n_ch_in, lam=0, reg_type='ridge'):
    n_ch_out = xy.shape[0]
    len_trf = xy.shape[1] / n_ch_in
    if reg_type == 'ridge':
        reg = np.eye(xxt.shape[0])
    elif reg_type == 'laplacian':
        reg = (np.diag(np.tile(np.concatenate(([1], 2 * np.ones(len_trf - 2),
                                               [1])), n_ch_in)) +
               np.diag(np.tile(np.concatenate((-np.ones(len_trf - 1), [0])),
                               n_ch_in)[:-1], 1) +
               np.diag(np.tile(np.concatenate((-np.ones(len_trf - 1), [0])),
                               n_ch_in)[:-1], -1))
    else:
        ValueError("reg_type must be either 'ridge' or 'laplacian'")

    w = np.dot(pinv(xxt + lam * reg), xy.T).T
    w = w.reshape([n_ch_out, n_ch_in, len_trf])

    return w
