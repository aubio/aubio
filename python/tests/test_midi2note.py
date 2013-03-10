#! /usr/bin/env python
# -*- coding: utf-8 -*-

from aubio import midi2note
import unittest

list_of_known_midis = (
        ( 0, 'C-1' ),
        ( 1, 'C#-1' ),
        ( 38, 'D2' ),
        ( 48, 'C3' ),
        ( 59, 'B3' ),
        ( 60, 'C4' ),
        ( 127, 'G9' ),
        )

class midi2note_good_values(unittest.TestCase):

    def test_midi2note_known_values(self):
        " known values are correctly converted "
        for midi, note in list_of_known_midis:
            self.assertEqual ( midi2note(midi), note )

class midi2note_wrong_values(unittest.TestCase):

    def test_midi2note_negative_value(self):
        " fails when passed a negative value "
        self.assertRaises(ValueError, midi2note, -2)

    def test_midi2note_negative_value(self):
        " fails when passed a value greater than 127 "
        self.assertRaises(ValueError, midi2note, 128)

    def test_midi2note_floating_value(self):
        " fails when passed a floating point "
        self.assertRaises(TypeError, midi2note, 69.2)

    def test_midi2note_character_value(self):
        " fails when passed a value that can not be transformed to integer "
        self.assertRaises(TypeError, midi2note, "a")

if __name__ == '__main__':
    unittest.main()
