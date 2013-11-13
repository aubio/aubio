# -*- coding: utf-8 -*-

def note2midi(note):
    " convert note name to midi note number, e.g. [C-1, G9] -> [0, 127] "
    _valid_notenames = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    _valid_modifiers = {None: 0, u'♮': 0, '#': +1, u'♯': +1, u'\udd2a': +2, 'b': -1, u'♭': -1, u'\ufffd': -2}
    _valid_octaves = range(-1, 10)
    if type(note) not in (str, unicode):
        raise TypeError, "a string is required, got %s" % note
    if not (1 < len(note) < 5):
        raise ValueError, "string of 2 to 4 characters expected, got %d (%s)" % (len(note), note)
    notename, modifier, octave = [None]*3

    if len(note) == 4:
        notename, modifier, octave_sign, octave = note
        octave = octave_sign + octave
    elif len(note) == 3:
        notename, modifier, octave = note
        if modifier == '-':
            octave = modifier + octave
            modifier = None
    else:
        notename, octave = note

    notename = notename.upper()
    octave = int(octave)

    if notename not in _valid_notenames:
        raise ValueError, "%s is not a valid note name" % notename
    if modifier not in _valid_modifiers:
        raise ValueError, "%s is not a valid modifier" % modifier
    if octave not in _valid_octaves:
        raise ValueError, "%s is not a valid octave" % octave

    midi = 12 + octave * 12 + _valid_notenames[notename] + _valid_modifiers[modifier]
    if midi > 127:
        raise ValueError, "%s is outside of the range C-2 to G8" % note
    return midi

def midi2note(midi):
    " convert midi note number to note name, e.g. [0, 127] -> [C-1, G9] "
    if type(midi) != int:
        raise TypeError, "an integer is required, got %s" % midi
    if not (-1 < midi < 128):
        raise ValueError, "an integer between 0 and 127 is excepted, got %d" % midi
    midi = int(midi)
    _valid_notenames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return _valid_notenames[midi % 12] + str( midi / 12 - 1)

def freq2note(freq):
    from aubio import freqtomidi
    return midi2note(int(freqtomidi(freq)))
