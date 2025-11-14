aubio-ledfx
===========

> **Note:** This is a maintained fork of the original [aubio project](https://github.com/aubio/aubio) by Paul Brossier, maintained by the [LedFx](https://github.com/LedFx) team.
>
> **Why this fork exists:**
> - The original aubio project is no longer actively maintained with regular releases
> - We provide Python 3.10-3.14 support with pre-built wheels on PyPI
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

This fork supports **Python 3.8 through 3.14** on Linux (x86_64, ARM64), macOS (Intel, Apple Silicon), and Windows (AMD64).

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

The core of aubio is written in C for portability and speed. The **pre-built wheels on PyPI** include the following optional features:

**All platforms:**
- [NumPy] integration for efficient array processing
- [libsndfile] for reading/writing uncompressed audio (WAV, AIFF, etc.)
- [fftw3] for fast Fourier transforms
- [libsamplerate] for high-quality audio resampling
- Audio codec support: FLAC, Vorbis/Ogg
- Built-in WAV reader/writer

**Platform-specific features:**
- **macOS:** [Accelerate] framework (optimized FFT), [CoreAudio] (native media reading), [ffmpeg], [rubberband] (time-stretching)
- **Windows:** [ffmpeg], [rubberband] (time-stretching) - all DLLs bundled in wheel
- **Linux:** MP3 support (mpg123, lame), Opus codec - static linking for portability

**Not included:** JACK audio, Intel IPP, BLAS/Atlas (for custom builds, see [building from source][doc_building])

For a detailed breakdown of features by platform, see the [Pre-built Wheel Features](#pre-built-wheel-features) appendix below.

[ffmpeg]: https://ffmpeg.org
[avcodec]: https://libav.org
[libsndfile]: http://www.mega-nerd.com/libsndfile/
[libsamplerate]: http://www.mega-nerd.com/SRC/
[CoreAudio]: https://developer.apple.com/reference/coreaudio
[Atlas]: http://math-atlas.sourceforge.net/
[Blas]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[fftw3]: http://fftw.org
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
| **Audio File I/O** | | | | |
| libsndfile | ✅ | ✅ | ✅ | Read/write uncompressed audio (WAV, AIFF, AU, etc.) |
| ffmpeg/libav | ❌ | ✅ | ✅ | Decode almost any media format (MP4, MKV, WebM, etc.) |
| CoreAudio | — | ✅ | — | Native macOS/iOS audio file reading (all Apple formats) |
| Built-in WAV | ✅ | ✅ | ✅ | Simple WAV support without external libraries |
| **Audio Codecs** | | | | |
| FLAC | ✅ | ✅ | ✅ | FLAC lossless audio codec |
| Vorbis/Ogg | ✅ | ✅ | ✅ | Ogg Vorbis lossy audio codec |
| MP3 (mpg123) | ✅ | ❌ | ❌ | MP3 decoding (Linux only) |
| MP3 (lame) | ✅ | ❌ | ❌ | MP3 encoding (Linux only) |
| Opus | ✅ | ❌ | ❌ | Opus low-latency codec (Linux only) |
| **Sample Rate Conversion** | | | | |
| libsamplerate | ✅ | ✅ | ✅ | High-quality audio resampling (SRC) |
| **Time Stretching** | | | | |
| rubberband | ❌ | ✅ | ✅ | Audio time-stretching and pitch-shifting |
| **FFT Implementation** | | | | |
| fftw3f | ✅ | ✅ | ✅ | Fast Fourier Transform (single precision, recommended) |
| Accelerate | — | ✅ | — | Apple's optimized FFT and DSP framework |
| ooura | ✅ | ✅ | ✅ | Fallback FFT implementation (always included) |

### Platform-Specific Notes

**Linux (x86_64, ARM64):**
- All dependencies are **statically linked** for maximum portability
- No external `.so` files required - works on any manylinux-compatible system
- Excludes rubberband and ffmpeg due to static linking constraints
- Includes MP3 and Opus codecs as transitive dependencies of libsndfile

**macOS (Intel x86_64, Apple Silicon ARM64):**
- Uses native **Accelerate framework** for optimized FFT operations
- Uses **CoreAudio** for reading all macOS-supported media formats
- Includes rubberband and ffmpeg for maximum format compatibility
- Separate wheel builds for Intel and Apple Silicon architectures
- Minimum deployment target: macOS 10.15 (Intel), macOS 11.0 (Apple Silicon)

**Windows (AMD64):**
- All dependency DLLs are **bundled inside the wheel** via delvewheel
- Fully portable - no separate installation of dependencies required
- Includes rubberband and ffmpeg
- Works on Windows 10+ (x64)

### Features NOT Included in Wheels

The following optional features are **not included** in pre-built wheels but can be enabled when [building from source][doc_building]:

- **JACK audio server:** Real-time audio I/O (requires system JACK installation)
- **Intel IPP:** Intel's performance primitives (commercial license required)
- **BLAS/Atlas:** Linear algebra acceleration (minimal benefit for aubio's use cases)
- **Double precision mode:** Single precision (float32) is used by default

### Python Version Support

Pre-built wheels are available for:
- **Python 3.10, 3.11, 3.12, 3.13, 3.14**
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
