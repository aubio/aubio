"""Synthetic test signals with exact ground truth.

Every generator is seeded and returns float64 samples in [-1, 1] together
with the times (in seconds) of what an analysis should find: beats, onsets,
note pitches, tempo. Hits are placed at the nearest sample to their exact
time, and their times are reported as placed.
"""

import numpy as np


class Signal:
    """samples plus ground truth"""

    def __init__(self, name, samplerate, x, beats=None, onsets=None,
                 tempo=None, notes=None):
        self.name = name
        self.samplerate = samplerate
        self.x = x
        # beat times, in seconds
        self.beats = np.asarray(beats if beats is not None else [], float)
        # onset times, in seconds
        self.onsets = np.asarray(onsets if onsets is not None else [], float)
        # tempo(t) -> bpm at time t (seconds), vectorised; None if none
        self.tempo = tempo
        # (start, end, frequency in Hz) of each steady note
        self.notes = notes if notes is not None else []

    @property
    def duration(self):
        return len(self.x) / self.samplerate


def _kick(sr):
    # pitch falls from 150 Hz to ~50 Hz: integrate the frequency for the
    # phase, sin(2 pi f(t) t) would be a different, wrong chirp
    t = np.arange(int(0.3 * sr)) / sr
    freq = 50 + 100 * np.exp(-30 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-8 * t)


def _snare(sr, rng):
    t = np.arange(int(0.15 * sr)) / sr
    noise = rng.standard_normal(t.size) * np.exp(-15 * t)
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-20 * t)
    return 0.5 * noise + 0.5 * tone


def _hihat(sr, rng):
    t = np.arange(int(0.05 * sr)) / sr
    # first difference of white noise: a high-passed burst
    noise = np.diff(rng.standard_normal(t.size + 1))
    return noise * np.exp(-60 * t)


def _place(x, sound, time, sr, gain):
    start = int(round(time * sr))
    if start >= x.size:
        return None
    end = min(start + sound.size, x.size)
    x[start:end] += gain * sound[:end - start]
    return start / sr


def drums(beat_times, duration, sr=44100, seed=1, noise=0.01, name="drums"):
    """A rock beat on the given beat times: kick on 1 and 3, snare on 2
    and 4, hi-hat on every eighth note. Every beat is a beat; every hit is
    an onset."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(duration * sr))
    kick, snare = _kick(sr), _snare(sr, rng)
    beats, onsets = [], []
    beat_times = np.asarray(beat_times, float)
    for i, bt in enumerate(beat_times):
        drum = (kick, 1.0) if i % 2 == 0 else (snare, 0.7)
        placed = _place(x, drum[0], bt, sr, drum[1])
        if placed is None:
            break
        _place(x, _hihat(sr, rng), bt, sr, 0.25)
        beats.append(placed)
        onsets.append(placed)
        # off-beat hi-hat, half way to the next beat
        nxt = beat_times[i + 1] if i + 1 < beat_times.size else bt + (
            bt - beat_times[i - 1] if i else 0.5)
        off = _place(x, _hihat(sr, rng), (bt + nxt) / 2, sr, 0.25)
        if off is not None:
            onsets.append(off)
    x += noise * rng.standard_normal(x.size)
    x *= 0.9 / np.max(np.abs(x))
    return beats, onsets, x


def steady_drums(bpm, duration=20.0, sr=44100, seed=1):
    period = 60.0 / bpm
    times = np.arange(0.5, duration, period)
    beats, onsets, x = drums(times, duration, sr, seed)
    return Signal("drums%d" % bpm, sr, x, beats, onsets,
                  tempo=lambda t: np.full(np.shape(t), float(bpm)))


SECTIONS = [(120, 10.0), (140, 10.0), (100, 10.0), (160, 10.0), (80, 10.0),
            (120, 10.0)]


def tempo_sections(sections=SECTIONS, sr=44100, seed=2):
    """Steady sections whose tempo jumps, from PR #34's benchmark: the
    tracker should follow each change and lock again."""
    times, start = [], 0.0
    edges = [0.0]
    for bpm, length in sections:
        period = 60.0 / bpm
        t = start + 0.5 if not times else times[-1] + period
        while t < start + length:
            times.append(t)
            t += period
        start += length
        edges.append(start)
    beats, onsets, x = drums(times, start, sr, seed)
    bpms = np.array([b for b, _ in sections], float)
    edges = np.array(edges)

    def tempo(t):
        i = np.clip(np.searchsorted(edges, t, side="right") - 1, 0,
                    bpms.size - 1)
        return bpms[i]
    sig = Signal("sections", sr, x, beats, onsets, tempo=tempo)
    sig.edges = edges
    return sig


def accelerando(bpm0=90.0, bpm1=150.0, duration=30.0, sr=44100, seed=3):
    """Tempo ramping linearly from bpm0 to bpm1. Beats are where the beat
    phase, the integral of bpm / 60, crosses an integer (PR #34's
    'gradual' file held 120 bpm throughout)."""
    slope = (bpm1 - bpm0) / duration
    # phase(t) = (bpm0 t + slope t^2 / 2) / 60 = k: solve for t
    times, k = [], 1
    while True:
        if slope == 0:
            t = 60.0 * k / bpm0
        else:
            t = (-bpm0 + np.sqrt(bpm0 ** 2 + 2 * slope * 60.0 * k)) / slope
        if t >= duration:
            break
        times.append(t)
        k += 1
    beats, onsets, x = drums(times, duration, sr, seed)
    return Signal("accelerando", sr, x, beats, onsets,
                  tempo=lambda t: bpm0 + slope * np.asarray(t))


NOTES = [220.0, 330.0, 440.0, 660.0, 880.0, 523.25, 261.63, 392.0]


def notes(freqs=NOTES, length=0.75, sr=44100, seed=4, harmonics=3):
    """Harmonic notes with short attacks and releases, a little noise:
    every note start is an onset, every steady part has a known pitch."""
    rng = np.random.default_rng(seed)
    n = int(length * sr)
    attack, release = int(0.01 * sr), int(0.05 * sr)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    t = np.arange(n) / sr
    parts, onsets, steady = [], [], []
    for i, f in enumerate(freqs):
        note = sum(np.sin(2 * np.pi * f * h * t + rng.uniform(0, 2 * np.pi))
                   / h for h in range(1, harmonics + 1))
        parts.append(0.4 * env * note)
        start = i * length
        onsets.append(start)
        steady.append((start + 0.1, start + length - 0.1, f))
    x = np.concatenate(parts + [np.zeros(int(0.5 * sr))])
    x += 0.001 * rng.standard_normal(x.size)
    return Signal("notes", sr, x, onsets=onsets, notes=steady)


def sweep(f0=60.0, f1=4000.0, duration=6.0, sr=44100):
    t = np.arange(int(duration * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2))
    return Signal("sweep", sr, x)


def noise(duration=6.0, sr=44100, seed=5):
    rng = np.random.default_rng(seed)
    x = 0.3 * rng.standard_normal(int(duration * sr))
    return Signal("noise", sr, np.clip(x, -1, 1))


def all_signals(sr=44100):
    """the signals the record and accuracy modes use, at samplerate sr"""
    return [
        sweep(sr=sr), noise(sr=sr), notes(sr=sr),
        steady_drums(120, sr=sr), steady_drums(160, sr=sr),
        tempo_sections(sr=sr), accelerando(sr=sr),
    ]
