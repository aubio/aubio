#! /usr/bin/env python

import sys
from aubio import pvoc, source
from numpy import array, arange, zeros, shape, log10, vstack
from pylab import imshow, show, cm, axis, ylabel, xlabel, xticks, yticks

def get_spectrogram(filename):
  samplerate = 44100
  win_s = 512                                        # fft window size
  hop_s = win_s / 2                                  # hop size
  fft_s = win_s / 2 + 1                              # spectrum bins

  a = source(filename, samplerate, hop_s)            # source file
  pv = pvoc(win_s, hop_s)                            # phase vocoder
  specgram = zeros([0, fft_s], dtype='float32')      # numpy array to store spectrogram

  # analysis
  while True:
    samples, read = a()                              # read file
    specgram = vstack((specgram,pv(samples).norm))   # store new norm vector
    if read < a.hop_size: break

  # plotting
  imshow(log10(specgram.T + .001), origin = 'bottom', aspect = 'auto', cmap=cm.gray_r)
  axis([0, len(specgram), 0, len(specgram[0])])
  # show axes in Hz and seconds
  time_step = hop_s / float(samplerate)
  total_time = len(specgram) * time_step
  print "total time: %0.2fs" % total_time,
  print ", samplerate: %.2fkHz" % (samplerate / 1000.)
  n_xticks = 10
  n_yticks = 10
  xticks_pos = [          x / float(n_xticks) * len(specgram) for x in range(n_xticks) ]
  xticks_str = [  "%.2f" % (x * total_time / float(n_xticks)) for x in range(n_xticks) ]
  xticks( xticks_pos , xticks_str )
  yticks_pos = [           y / float(n_yticks) * len(specgram[0]) for y in range(n_yticks) ]
  yticks_str = [ "%.2f" % (y * samplerate / 2000. / float(n_yticks)) for y in range(n_yticks) ]
  yticks( yticks_pos , yticks_str )
  ylabel('Frequency (kHz)')
  xlabel('Time (s)')

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "Usage: %s <filename>" % sys.argv[0]
  else:
    for soundfile in sys.argv[1:]:
      get_spectrogram(soundfile)
      # display graph
      show()
