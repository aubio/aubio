#! /usr/bin/env python

from aubio import filterbank, fvec
from pylab import loglog, show, xlim, ylim, xlabel, ylabel, title
from numpy import vstack, arange

win_s = 2048
samplerate = 48000

freq_list = [60, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 24000]
n_filters = len(freq_list) - 2

f = filterbank(n_filters, win_s)
freqs = fvec(freq_list)
f.set_triangle_bands(freqs, samplerate)

coeffs = f.get_coeffs()
coeffs[4] *= 5.

f.set_coeffs(coeffs)

times = vstack([arange(win_s // 2 + 1) * samplerate / win_s] * n_filters)
title('Bank of filters built using a simple list of boundaries\nThe middle band has been amplified by 2.')
loglog(times.T, f.get_coeffs().T, '.-')
xlim([50, samplerate/2])
ylim([1.0e-6, 2.0e-2])
xlabel('log frequency (Hz)')
ylabel('log amplitude')

show()
