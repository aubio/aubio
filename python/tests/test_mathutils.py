#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from aubio import bintomidi, miditobin, freqtobin, bintofreq, freqtomidi, miditofreq

class aubio_mathutils(TestCase):

    def test_miditobin(self):
       a = [ miditobin(a, 44100, 512) for a in range(128) ]

    def test_bintomidi(self):
       a = [ bintomidi(a, 44100, 512) for a in range(128) ]

    def test_freqtobin(self):
       a = [ freqtobin(a, 44100, 512) for a in range(128) ]

    def test_bintofreq(self):
       a = [ bintofreq(a, 44100, 512) for a in range(128) ]

    def test_freqtomidi(self):
       a = [ freqtomidi(a) for a in range(128) ]

    def test_miditofreq(self):
       freqs = [ miditofreq(a) for a in range(128) ]
       midis = [ freqtomidi(a) for a in freqs ]
       print midis

if __name__ == '__main__':
    from unittest import main
    main()
