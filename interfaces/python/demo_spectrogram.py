#! /usr/bin/python

import sys
from aubio import pvoc, source
from numpy import array, arange, zeros, shape, log10, vstack
from pylab import imshow, show, gray, autumn, axis, ylabel, xlabel, xticks, yticks

def get_spectrogram(filename):
  win_s = 512                 # fft size
  hop_s = win_s / 2           # hop size
  fft_s = win_s / 2 + 1       # number of spectrum bins
  samplerate = 16000
  specgram = zeros([0, fft_s], dtype='float32')
  a = source(filename, samplerate, hop_s)                 # mono 8kHz only please
  pv = pvoc(win_s, hop_s)                            # phase vocoder
  while True:
    samples, read = a()                              # read file
    #specgram = vstack((specgram,1.-log10(1.+pv(samples).norm)))   # store new norm vector
    specgram = vstack((specgram,pv(samples).norm))   # store new norm vector
    if read < a.hop_size: break

  autumn()
  from pylab import gray
  gray()
  imshow(specgram.T, origin = 'bottom', aspect = 'auto')
  axis([0, len(specgram), 0, len(specgram[0])])
  ylabel('Frequency (Hz)')
  xlabel('Time (s)')
  time_step = hop_s / float(samplerate)
  total_time = len(specgram) * time_step
  ticks = 10
  xticks( arange(ticks) / float(ticks) * len(specgram),
      [x * total_time / float(ticks) for x in range(ticks) ] )
  yticks( arange(ticks) / float(ticks) * len(specgram[0]),
      [x * samplerate / 2. / float(ticks) for x in range(ticks) ] )
  show()

if len(sys.argv) < 2:
  print "Usage: %s <filename>" % sys.argv[0]
else:
  [get_spectrogram(soundfile) for soundfile in sys.argv[1:]]
