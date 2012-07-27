#! /usr/bin/python

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
  imshow(log10(specgram.T), origin = 'bottom', aspect = 'auto', cmap=cm.gray_r)
  axis([0, len(specgram), 0, len(specgram[0])])
  ylabel('Frequency (Hz)')
  xlabel('Time (s)')
  # show axes in Hz and seconds
  time_step = hop_s / float(samplerate)
  total_time = len(specgram) * time_step
  ticks = 10
  xticks( arange(ticks) / float(ticks) * len(specgram),
      [x * total_time / float(ticks) for x in range(ticks) ] )
  yticks( arange(ticks) / float(ticks) * len(specgram[0]),
      [x * samplerate / 2. / float(ticks) for x in range(ticks) ] )

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "Usage: %s <filename>" % sys.argv[0]
  else:
    for soundfile in sys.argv[1:]:
      get_spectrogram(soundfile)
      # display graph
      show()
