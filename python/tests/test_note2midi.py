# -*- encoding: utf8 -*-

from aubio import note2midi
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
        ( 'G8', 115 ),
        ( u'G♯8', 116 ),
        ( u'G♭9', 126 ),
        ( 'G9', 127 ),
        ( u'G\udd2a2', 45 ),
        ( u'B\ufffd2', 45 ),
        ( u'A♮2', 45 ),
        )

class TestNote2MidiGoodValues(unittest.TestCase):

    def test_note2midi_known_values(self):
        " known values are correctly converted "
        for note, midi in list_of_known_notes:
            self.assertEqual ( note2midi(note), midi )

class TestNote2MidiWrongValues(unittest.TestCase):

    def test_note2midi_missing_octave(self):
        " fails when passed only one character"
        self.assertRaises(ValueError, note2midi, 'C')

    def test_note2midi_wrong_modifier(self):
        " fails when passed an invalid note name"
        self.assertRaises(ValueError, note2midi, 'C.1')

    def test_note2midi_wronge_midifier_again(self):
        " fails when passed a wrong modifier"
        self.assertRaises(ValueError, note2midi, 'CB-3')

    def test_note2midi_wrong_octave(self):
        " fails when passed a wrong octave"
        self.assertRaises(ValueError, note2midi, 'CBc')

    def test_note2midi_out_of_range(self):
        " fails when passed a non-existing note"
        self.assertRaises(ValueError, note2midi, 'A9')

if __name__ == '__main__':
    unittest.main()
