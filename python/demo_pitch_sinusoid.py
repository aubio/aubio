#! /usr/bin/env python

from numpy import random, sin, arange, ones, zeros
from math import pi
from aubio import fvec, pitch

def build_sinusoid(length, freqs, samplerate):
  return sin( 2. * pi * arange(length) * freqs / samplerate)

def run_pitch(p, input_vec):
  f = fvec (p.hop_size)
  cands = []
  count = 0
  for vec_slice in input_vec.reshape((-1, p.hop_size)):
    f[:] = vec_slice
    cands.append(p(f))
  return cands

methods = ['default', 'schmitt', 'fcomb', 'mcomb', 'yin', 'yinfft']

cands = {}
buf_size = 2048
hop_size = 512
samplerate = 44100
sin_length = (samplerate * 10) % 512 * 512
freqs = zeros(sin_length)

partition = sin_length / 8
pointer = 0

pointer += partition
freqs[pointer: pointer + partition] = 440

pointer += partition
pointer += partition
freqs[ pointer : pointer + partition ] = 740

pointer += partition
freqs[ pointer : pointer + partition ] = 1480

pointer += partition
pointer += partition
freqs[ pointer : pointer + partition ] = 400 + 5 * random.random(sin_length/8)

a = build_sinusoid(sin_length, freqs, samplerate)

for method in methods:
  p = pitch(method, buf_size, hop_size, samplerate)
  cands[method] = run_pitch(p, a)

print "done computing"

if 1:
  from pylab import plot, show, xlabel, ylabel, legend, ylim
  ramp = arange(0, sin_length / hop_size).astype('float') * hop_size / samplerate
  for method in methods:
    plot(ramp, cands[method],'.-')

  # plot ground truth
  ramp = arange(0, sin_length).astype('float') / samplerate
  plot(ramp, freqs, ':')

  legend(methods+['ground truth'], 'upper right')
  xlabel('time (s)')
  ylabel('frequency (Hz)')
  ylim([0,2000])
  show()

