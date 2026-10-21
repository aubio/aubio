aubio-ledfx
===========

> **Note:** This is a maintained fork of the original [aubio project](https://github.com/aubio/aubio) by Paul Brossier, maintained by the [LedFx](https://github.com/LedFx) team.
>
> **Why this fork exists:**
> - The original aubio project is no longer actively maintained with regular releases
> - We provide Python 3.11-3.15 support with pre-built wheels on PyPI
> - This fork includes the latest fixes and improvements from aubio's main branch
> - LedFx and other projects depend on aubio and need a reliable, up-to-date package
>
> **All credit for aubio goes to the original author Paul Brossier and contributors.**
>
> **Original project:** https://github.com/aubio/aubio
> **This fork:** https://github.com/LedFx/aubio-ledfx
> **PyPI package:** https://pypi.org/project/aubio-ledfx/

---

## About aubio

aubio is a collection of tools for music and audio analysis.

This package integrates the aubio library with [NumPy] to provide a set of
efficient tools to process and analyse audio signals, including:

- read audio from any media file, including videos and remote streams
- high quality phase vocoder, spectral filterbanks, and linear filters
- Mel-Frequency Cepstrum Coefficients and standard spectral descriptors
- detection of note attacks (onset)
- pitch tracking (fundamental frequency estimation)
- beat detection and tempo tracking

This fork supports **Python 3.11 through 3.15** on Linux (x86_64, ARM64), macOS (Intel, Apple Silicon), and Windows (AMD64).

Installation
------------

Install from PyPI:

```bash
pip install aubio-ledfx
```

Pre-built wheels are available for:
- **Linux:** x86_64, ARM64 (manylinux)
- **macOS:** Intel (x86_64), Apple Silicon (ARM64)
- **Windows:** AMD64

Links
-----

- [PyPI package][pypi]
- [module documentation][doc_python]
- [installation instructions][doc_python_install]
- [aubio manual][manual]
- [original aubio homepage][homepage]
- [issue tracker (this fork)][bugtracker]

Demos
-----

Some examples are available in the [`python/demos` folder][demos_dir]. Each
script is a command line program which accepts one ore more argument.

**Notes**: installing additional modules is required to run some of the demos.

### Analysis

- `demo_source.py` uses aubio to read audio samples from media files
- `demo_onset_plot.py` detects attacks in a sound file and plots the results
  using [matplotlib]
- `demo_pitch.py` looks for fundamental frequency in a sound file and plots the
  results using [matplotlib]
- `demo_spectrogram.py`, `demo_specdesc.py`, `demo_mfcc.py` for spectral
  analysis.

### Real-time

- `demo_pyaudio.py` and `demo_tapthebeat.py` use [pyaudio]
- `demo_pysoundcard_play.py`, `demo_pysoundcard.py` use [PySoundCard]
- `demo_alsa.py` uses [pyalsaaudio]

### Others

- `demo_timestretch.py` can change the duration of an input file and write the
  new sound to disk,
- `demo_wav2midi.py` detects the notes in a file and uses [mido] to write the
  results into a MIDI file

### Example

Use `demo_timestretch_online.py` to slow down `loop.wav`, write the results in
`stretched_loop.wav`:

    $ python demo_timestretch_online.py loop.wav stretched_loop.wav 0.92

Building from Source
--------------------

This fork uses the [Meson build system](https://mesonbuild.com/) and [vcpkg](https://vcpkg.io) for dependency management.

### Quick Build

```bash
# Install build dependencies
pip install "meson>=1.9.0" meson-python ninja numpy

# Build and install
pip install .
```

For detailed build instructions, see the [main README](https://github.com/LedFx/aubio-ledfx#readme).

Built with
----------

The core of aubio is written in C for portability and speed. The **pre-built
wheels on PyPI** are built with every optional feature that works in a Python
package, the same on every platform:

- [NumPy] integration for efficient array processing
- [ffmpeg] for reading almost any audio or video file (MP4/AAC, MKV, WebM, ...)
- [libsndfile] for reading and writing WAV, AIFF, FLAC, Ogg Vorbis, Opus and MP3
- FLAC and Ogg Vorbis sinks, and a built-in WAV reader and writer
- [libsamplerate] for high-quality resampling
- [rubberband] for time-stretching and pitch-shifting
- on macOS, the [Accelerate] framework (FFT and vector maths) and [CoreAudio]
  (every format macOS can read)

**Not included:** JACK (only aubio's command-line tools use it), Intel IPP
and BLAS (see the appendix, and [building from source][doc_building]).

For a detailed breakdown, see the [Pre-built Wheel Features](#pre-built-wheel-features) appendix below.

[ffmpeg]: https://ffmpeg.org
[avcodec]: https://libav.org
[libsndfile]: http://www.mega-nerd.com/libsndfile/
[libsamplerate]: http://www.mega-nerd.com/SRC/
[CoreAudio]: https://developer.apple.com/reference/coreaudio
[Atlas]: http://math-atlas.sourceforge.net/
[Blas]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[fftw3]: http://fftw.org
[rubberband]: https://breakfastquay.com/rubberband/
[Accelerate]: https://developer.apple.com/reference/accelerate
[Intel IPP]: https://software.intel.com/en-us/intel-ipp

[demos_dir]:https://github.com/aubio/aubio/tree/master/python/demos
[pyaudio]:https://people.csail.mit.edu/hubert/pyaudio/
[PySoundCard]:https://github.com/bastibe/PySoundCard
[pyalsaaudio]:https://larsimmisch.github.io/pyalsaaudio/
[mido]:https://mido.readthedocs.io

---

## Pre-built Wheel Features

This appendix provides a complete breakdown of which optional features are included in the pre-built wheels distributed on PyPI.

### Features by Platform

| Feature | Linux | macOS | Windows | Description |
|---------|:-----:|:-----:|:-------:|-------------|
| **Audio file input and output** | | | | |
| ffmpeg/libav | ✅ | ✅ | ✅ | Decode almost any media format (MP4/AAC, MKV, WebM, ...) |
| libsndfile | ✅ | ✅ | ✅ | WAV, AIFF, AU and more, with the codecs below |
| CoreAudio | — | ✅ | — | Native macOS reading and writing (every Apple format) |
| Built-in WAV | ✅ | ✅ | ✅ | WAV reader and writer without external libraries |
| **Codecs (through libsndfile)** | | | | |
| FLAC | ✅ | ✅ | ✅ | Read and write, plus aubio's own FLAC sink |
| Ogg Vorbis | ✅ | ✅ | ✅ | Read and write, plus aubio's own Vorbis sink |
| Opus | ✅ | ✅ | ✅ | Read and write |
| MP3 (mpg123, LAME) | ✅ | ✅ | ✅ | Read and write |
| **Resampling and effects** | | | | |
| libsamplerate | ✅ | ✅ | ✅ | High-quality resampling |
| rubberband | ✅ | ✅ | ✅ | Time-stretching and pitch-shifting |
| **FFT** | | | | |
| Accelerate (vDSP) | — | ✅ | — | Apple's FFT and vector maths |
| ooura | ✅ | — | ✅ | aubio's built-in FFT (power-of-two sizes) |

Each wheel build prints meson's feature summary in its CI log, so this table
can be checked against any release's build.

### Platform-Specific Notes

**Linux (x86_64, ARM64, manylinux_2_28):**
- Every dependency is **statically linked** into the extension, so the wheel
  needs no other `.so` files.

**macOS (Intel x86_64, Apple Silicon ARM64):**
- Dependencies are statically linked.
- Minimum macOS: 10.15 (Intel), 11.0 (Apple Silicon).

**Windows (AMD64):**
- Built with MSVC. The dependency DLLs, and the C++ runtime (`msvcp140.dll`)
  rubberband was built against, are **bundled inside the wheel** by delvewheel.

### Features NOT Included in Wheels

These can be enabled when [building from source][doc_building]:

- **JACK:** only aubio's command-line tools use it, and the wheels don't ship
  them. From Python, read live audio with `sounddevice` or PyAudio and pass the
  blocks to aubio.
- **FFTW:** `-Dfftw3f=enabled` adds non-power-of-two FFT sizes. aubio's own
  ooura FFT measured as fast or faster at the usual power-of-two sizes.
- **Intel IPP** (Intel's proprietary libraries) and **BLAS:** speed-ups only,
  with no new features.
- **Double precision:** the wheels use single precision (float32).

### Python Version Support

Pre-built wheels are available for:
- **CPython 3.11, 3.12, 3.13, 3.14, 3.15**
- All wheels include the same feature set per platform

---

[pypi]: https://pypi.org/project/aubio-ledfx/
[manual]: https://aubio.org/manual/latest/
[doc_python]: https://aubio.org/manual/latest/python.html
[doc_python_install]: https://aubio.org/manual/latest/python_module.html
[doc_building]: https://github.com/LedFx/aubio-ledfx#quick-start---building-aubio
[homepage]: https://aubio.org
[NumPy]: https://www.numpy.org
[bugtracker]: https://github.com/LedFx/aubio-ledfx/issues
[matplotlib]:https://matplotlib.org/
