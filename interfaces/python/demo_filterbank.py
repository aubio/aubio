from _aubio import filterbank
from numpy import array

f = filterbank(9, 1024)
freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
freqs = array(freq_list, dtype = 'float32')
f.set_triangle_bands(freqs, 48000)
f.get_coeffs().T

from pylab import loglog, show
loglog(f.get_coeffs().T, '+-')
show()

