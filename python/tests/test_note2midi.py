#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from aubio import note2midi, freq2note
from nose2.tools import params
import unittest

list_of_known_notes = (
        ( 'C-1', 0 ),
        ( 'C#-1', 1 ),
        ( 'd2', 38 ),
        ( 'C3', 48 ),
        ( 'B3', 59 ),
        ( 'B#3', 60 ),
        ( 'C♯4', 61 ),
        ( 'A4', 69 ),
        ( 'A#4', 70 ),
        ( 'A♯4', 70 ),
        ( 'A\u266f4', 70 ),
        ( 'Bb4', 70 ),
        ( 'B♭4', 70 ),
        ( 'B\u266d4', 70 ),
        ( 'G8', 115 ),
        ( 'G♯8', 116 ),
        ( 'G9', 127 ),
        ( 'A♮2', 45 ),
        )

list_of_known_notes_with_unicode_issues = (
        ('C𝄪4', 62 ),
        ('E𝄫4', 62 ),
        )

list_of_unknown_notes = (
        ( 'G\udd2a2' ),
        ( 'B\ufffd2' ),
        ( 'B\u266e\u266e2' ),
        ( 'B\u266f\u266d3' ),
        ( 'B33' ),
        ( 'C.3' ),
        ( 'A' ),
        ( '2' ),
        )

class note2midi_good_values(unittest.TestCase):

    @params(*list_of_known_notes)
    def test_note2midi_known_values(self, note, midi):
        " known values are correctly converted "
        self.assertEqual ( note2midi(note), midi )

    @params(*list_of_known_notes_with_unicode_issues)
    def test_note2midi_known_values_with_unicode_issues(self, note, midi):
        " known values are correctly converted, unless decoding is expected to fail"
        try:
            self.assertEqual ( note2midi(note), midi )
        except UnicodeEncodeError as e:
            import sys
            strfmt = "len(u'\\U0001D12A') != 1, excpected decoding failure | {:s} | {:s} {:s}"
            strres = strfmt.format(e, sys.platform, sys.version)
            # happens with: darwin 2.7.10, windows 2.7.12
            if len('\U0001D12A') != 1 and sys.version[0] == '2':
                self.skipTest(strres + " | upgrade to Python 3 to fix")
            else:
                raise

class note2midi_wrong_values(unittest.TestCase):

    def test_note2midi_missing_octave(self):
        " fails when passed only one character"
        self.assertRaises(ValueError, note2midi, 'C')

    def test_note2midi_wrong_modifier(self):
        " fails when passed a note with an invalid modifier "
        self.assertRaises(ValueError, note2midi, 'C.1')

    def test_note2midi_another_wrong_modifier_again(self):
        " fails when passed a note with a invalid note name "
        self.assertRaises(ValueError, note2midi, 'CB-3')

    def test_note2midi_wrong_octave(self):
        " fails when passed a wrong octave number "
        self.assertRaises(ValueError, note2midi, 'CBc')

    def test_note2midi_out_of_range(self):
        " fails when passed a note out of range"
        self.assertRaises(ValueError, note2midi, 'A9')

    def test_note2midi_wrong_note_name(self):
        " fails when passed a note with a wrong name"
        self.assertRaises(ValueError, note2midi, 'W9')

    def test_note2midi_low_octave(self):
        " fails when passed a note with a too low octave"
        self.assertRaises(ValueError, note2midi, 'C-9')

    def test_note2midi_wrong_data_type(self):
        " fails when passed a non-string value "
        self.assertRaises(TypeError, note2midi, 123)

    def test_note2midi_wrong_data_too_long(self):
        " fails when passed a note with a note name longer than expected"
        self.assertRaises(ValueError, note2midi, 'CB+-3')

    @params(*list_of_unknown_notes)
    def test_note2midi_unknown_values(self, note):
        " unknown values throw out an error "
        self.assertRaises(ValueError, note2midi, note)

class freq2note_simple_test(unittest.TestCase):

    def test_freq2note(self):
        " make sure freq2note(441) == A4 "
        self.assertEqual("A4", freq2note(441))

if __name__ == '__main__':
    import nose2
    nose2.main()
