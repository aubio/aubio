# Numerical checks

Tools to see what a change does to aubio's numbers, and how well aubio
finds what it is meant to find. They need numpy and an installed aubio:
build the module, then run them with that Python.

```sh
# every output of every algorithm, on fixed signals, at 44100/512/2048 and
# LedFx's 30000/500/4096 (add --sounds python/tests/sounds for the test files)
python scripts/numerics/numerics.py record before.npz
# ... change something, rebuild, reinstall ...
python scripts/numerics/numerics.py record after.npz
python scripts/numerics/numerics.py compare before.npz after.npz
```

Inputs are float32 values whatever the build, so a double-precision build
(`-Ddouble=true`) of the same code records a reference. With `--ref`,
`compare` gives the error of each recording against it, and marks each
output better or worse. That tells a precision change that moves the
numbers closer to the true values from one that moves them away:

```sh
python scripts/numerics/numerics.py compare before.npz after.npz --ref double.npz
```

Onset times, beats and notes are compared as events, with an F-measure:
a threshold that flips moves an event, it doesn't make the output wrong.
The phase-based onset functions (phase, wphase, specdiff) are close to
their thresholds on noiseless tones, so their events change with any
rounding.

`accuracy` scores the algorithms against the generated signals' ground
truth (`signals.py`), with mir_eval's definitions:

- onsets: F-measure, within 50 ms
- beats: F-measure, within 70 ms, from 5 s on
- tempo: share of frames, from 5 s on, within 4% of the true tempo, and
  with half or double tempo allowed (`tempo/octave`)
- lock: after each tempo change, the time until the estimate is right
  (half or double allowed) and stays right for a second
- pitch: share of frames in the steady part of each note within 50 cents

```sh
python scripts/numerics/numerics.py accuracy --json scores.json
```

The signals are seeded, so every run is identical. They are a rock beat
at 120 and 160 bpm, the same beat in sections that jump between 120, 140,
100, 160, 80 and 120 bpm, an accelerando from 90 to 150 bpm, harmonic notes,
a sweep and noise. The beat and the sections come from PR #34's benchmark.
Its kick is now a real pitch drop, its noise is seeded, every beat is a
beat and every hit an onset, and the accelerando really changes tempo.
