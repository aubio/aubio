#! /usr/bin/env python

import sys
from aubio import pvoc, source
from numpy import array, arange, zeros, shape, log10, vstack
from pylab import imshow, show, cm, axis, ylabel, xlabel, xticks, yticks

def get_spectrogram(filename, samplerate = 0):
  win_s = 512                                        # fft window size
  hop_s = win_s / 2                                  # hop size
  fft_s = win_s / 2 + 1                              # spectrum bins

  a = source(filename, samplerate, hop_s)            # source file
  if samplerate == 0: samplerate = a.samplerate
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

  def get_rounded_ticks( top_pos, step, n_ticks ):
      top_label = top_pos * step
      # get the first label
      ticks_first_label = top_pos * step / n_ticks
      # round to the closest .1
      ticks_first_label = round ( ticks_first_label * 10. ) / 10.
      # compute all labels from the first rounded one
      ticks_labels = [ ticks_first_label * n for n in range(n_ticks) ] + [ top_label ]
      # get the corresponding positions
      ticks_positions = [ ticks_labels[n] / step for n in range(n_ticks) ] + [ top_pos ]
      # convert to string
      ticks_labels = [  "%.1f" % x for x in ticks_labels ]
      # return position, label tuple to use with x/yticks
      return ticks_positions, ticks_labels

  # apply to the axis
  xticks( *get_rounded_ticks ( len(specgram), time_step, n_xticks ) )
  yticks( *get_rounded_ticks ( len(specgram[0]), (samplerate / 2. / 1000.) / len(specgram[0]), n_yticks ) )
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
