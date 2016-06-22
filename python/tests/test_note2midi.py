#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from aubio import note2midi, freq2note
import unittest

list_of_known_notes = (
        ( 'C-1', 0 ),
        ( 'C#-1', 1 ),
        ( 'd2', 38 ),
        ( 'C3', 48 ),
        ( 'B3', 59 ),
        ( 'B#3', 60 ),
        ( 'A4', 69 ),
        ( 'A#4', 70 ),
        ( 'Bb4', 70 ),
        ( 'B♭4', 70 ),
        ( 'G8', 115 ),
        ( 'G♯8', 116 ),
        ( 'G9', 127 ),
        ( 'G\udd2a2', 45 ),
        ( 'B\ufffd2', 45 ),
        ( 'A♮2', 45 ),
        )

class note2midi_good_values(unittest.TestCase):

    def test_note2midi_known_values(self):
        " known values are correctly converted "
        for note, midi in list_of_known_notes:
            self.assertEqual ( note2midi(note), midi )

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


class freq2note_simple_test(unittest.TestCase):

    def test_freq2note(self):
        " make sure freq2note(441) == A4 "
        self.assertEqual("A4", freq2note(441))

if __name__ == '__main__':
    unittest.main()
