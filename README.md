aubio-ledfx
===========

[![Documentation](https://readthedocs.org/projects/aubio/badge/?version=latest)](http://aubio.readthedocs.io/en/latest/?badge=latest "Latest documentation")
[![DOI](https://zenodo.org/badge/396389.svg)](https://zenodo.org/badge/latestdoi/396389)

> **Note:** This is a fork of the original [aubio project](https://github.com/aubio/aubio) maintained by the [LedFx](https://github.com/LedFx) team.
>
> **Why this fork exists:**
> - The original aubio project is no longer actively maintained and released
> - We need Python 3.13 support with pre-built wheels on PyPI
> - We require the latest fixes and improvements from the main branch of aubio
> - LedFx depends on aubio and needs a reliable, up-to-date release
>
> All credit for aubio goes to the original authors. This fork exists solely to provide maintained releases for projects that depend on aubio.
>
> **Original project:** https://github.com/aubio/aubio
> **This fork:** https://github.com/LedFx/aubio-ledfx

---

## About aubio

aubio is a library to label music and sounds. It listens to audio signals and
attempts to detect events. For instance, when a drum is hit, at which frequency
is a note, or at what tempo is a rhythmic melody.

Its features include segmenting a sound file before each of its attacks,
performing pitch detection, tapping the beat and producing midi streams from
live audio.

aubio provide several algorithms and routines, including:

  - several onset detection methods
  - different pitch detection methods
  - tempo tracking and beat detection
  - MFCC (mel-frequency cepstrum coefficients)
  - FFT and phase vocoder
  - up/down-sampling
  - digital filters (low pass, high pass, and more)
  - spectral filtering
  - transient/steady-state separation
  - sound file read and write access
  - various mathematics utilities for music applications

The name aubio comes from _audio_ with a typo: some errors are likely to be
found in the results.

Python module
-------------

A python module for aubio is provided. For more information on how to use it,
please see the file [`python/README.md`](python/README.md) and the
[manual](https://aubio.org/manual/latest/) .

Tools
-----

The python module comes with the following command line tools:

 - `aubio` extracts informations from sound files
 - `aubiocut` slices sound files at onset or beat timestamps

Additional command line tools are included along with the library:

 - `aubioonset` outputs the time stamp of detected note onsets
 - `aubiopitch` attempts to identify a fundamental frequency, or pitch, for
   each frame of the input sound
 - `aubiomfcc` computes Mel-frequency Cepstrum Coefficients
 - `aubiotrack` outputs the time stamp of detected beats
 - `aubionotes` emits midi-like notes, with an onset, a pitch, and a duration
 - `aubioquiet` extracts quiet and loud regions

Documentation
-------------

  - [manual](https://aubio.org/manual/latest/), generated with sphinx
  - [developer documentation](https://aubio.org/doc/latest/), generated with Doxygen

The latest version of the documentation can be found at:

  https://aubio.org/documentation

Quick Start - Building aubio
----------------------------

aubio uses the [Meson build system](https://mesonbuild.com/) and [vcpkg](https://vcpkg.io) for dependency management. It compiles on Linux, macOS, Windows, and iOS.

### Prerequisites

- **Meson** >= 1.9.0: `pip install "meson>=1.9.0"` or `apt install meson`
- **Ninja**: `pip install ninja` or `apt install ninja-build`
- **C compiler**: GCC, Clang, or MSVC
- **Python** >= 3.8 (for Python bindings)
- **NumPy** (for Python bindings): `pip install numpy`
- **vcpkg** (optional, for automatic dependency management): See [vcpkg installation](https://vcpkg.io/en/getting-started.html)

### Dependency Management with vcpkg

aubio uses vcpkg manifest mode (vcpkg.json) to automatically manage dependencies. If vcpkg is installed, dependencies will be fetched automatically during build.

```bash
# Install vcpkg (one-time setup)
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh  # or .bat on Windows
export VCPKG_ROOT=$(pwd)/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

Dependencies (libsndfile, libsamplerate, fftw3) will be installed automatically via vcpkg.json when you build.

### Building the C Library

```bash
# Configure the build (vcpkg will install dependencies automatically)
meson setup builddir

# Compile
meson compile -C builddir

# Install (optional)
meson install -C builddir
```

### Building Python Wheels

aubio uses [meson-python](https://meson-python.readthedocs.io/) for Python packaging.

#### Using pip (recommended)

```bash
# Install from source directory
pip install .

# Or build a wheel
pip wheel . --no-deps
```

#### Using uv (faster alternative)

```bash
# Install uv if you haven't already
pip install uv

# Build a wheel
uv build --wheel

# Install the wheel
pip install dist/aubio-*.whl
```

### Build Options

Common build options (use with `meson setup -Doption=value builddir`):

- `-Dexamples=true/false` - Build example programs (default: false)
- `-Dtests=true/false` - Build test suite (default: false)
- `-Ddocs=true/false` - Build documentation (default: false)
- `-Ddouble=true/false` - Compile in double precision (default: false)

Example:
```bash
meson setup builddir -Dexamples=true -Dtests=true
meson compile -C builddir
meson test -C builddir
```

### Platform-Specific Notes

**Windows:** Requires MSVC or MinGW-w64. The shared library is not built on Windows by default (only static library and Python extension).

**macOS:** Automatically detects and uses the Accelerate framework for optimized FFT operations.

**Linux:** Install optional dependencies for additional features:
- `libsndfile-dev` - For audio file I/O
- `libsamplerate-dev` - For sample rate conversion
- `libfftw3-dev` - For FFTW3 FFT implementation
- `libjack-dev` - For JACK audio support

### Documentation

For comprehensive build documentation, see:
- [`doc/building.rst`](doc/building.rst) - Complete build guide
- [`doc/meson_reference.rst`](doc/meson_reference.rst) - Quick reference for common tasks

Citation
--------

Please use the DOI link above to cite this release in your publications. For
more information, see also the [about
page](https://aubio.org/manual/latest/about.html) in [aubio
manual](https://aubio.org/manual/latest/).

Homepage
--------

**Original aubio project:** https://aubio.org/
**This fork (aubio-ledfx):** https://github.com/LedFx/aubio-ledfx

License
-------

aubio is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This fork maintains the same license as the original aubio project.

Contributing
------------

Patches are welcome: please fork the latest git repository and create a feature
branch. Submitted requests should pass all continuous integration tests.

- **Pull request titles** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`fix(tempo): ...`, `feat: ...`, `ci: ...`). PRs are squash-merged with the
  title as the commit message, and a check comments with a suggestion when a
  title does not parse.
- **Lint** with [prek](https://github.com/j178/prek): `uvx prek run --all-files`
  (or `uvx prek install` for a git hook). CI runs the same hooks, and
  autofix.ci pushes whatever they can fix to the PR.
- **Tests**: `meson setup builddir -Dtests=true && meson test -C builddir` for
  the C library; `pytest python/tests` against an installed build for Python.
  Wheel builds run the Python suite on every platform and fail on any failure.
  A pull request builds what its changes can affect (the `Plan` job in
  `.github/workflows/build.yml` says why), and `main` builds everything.
- **Releases** are made by [release-please](https://github.com/googleapis/release-please):
  it keeps a release PR open with the next version and changelog, built from
  the `fix:`/`feat:` titles on `main`. Merging it tags the release, and CI
  publishes the wheels to PyPI and the GitHub release.
- **Dependencies** (GitHub Actions, prek hooks) are kept current by Renovate
  with the shared [LedFx policy](https://github.com/LedFx/renovate-config).

### GitHub Copilot Support

This repository includes automated environment setup for GitHub Copilot Coding Agent. The `.github/workflows/copilot-setup-steps.yml` file ensures that Copilot has all necessary build tools, dependencies, and testing infrastructure pre-installed. For details, see [`.github/COPILOT_SETUP_README.md`](.github/COPILOT_SETUP_README.md).

For optimization and modernization priorities, see:
- [**OPTIMIZATION_SUMMARY.md**](OPTIMIZATION_SUMMARY.md) - Quick reference guide
- [**OPTIMIZATION_ROADMAP.md**](OPTIMIZATION_ROADMAP.md) - Detailed implementation roadmap

### Verified production publication

Production keeps `build.yml` and environment `pypi`, with the SHA-pinned
[shared release transaction](https://github.com/LedFx/release-ci) with the
generated native wheel plan and existing `pyproject.toml` metadata. One queued job downloads both
`wheels-*` and `cibw-sdist` from the same tested run. The existing Meson/vcpkg
version guards, C tests, wheel/native dependency and bundled-library/license
contracts are unchanged. Publication targets production PyPI only; manual
workflow runs build and test without publishing.

Production uses a scoped App token plus caller OIDC and GitHub attestations to
validate archive metadata, source/tag, exact hashes and provenance before
publishing the existing release-please draft last. Softprops finalization is
replaced by explicit immutable release-ID finalization, preserving release
notes. Missing drafts/conflicting files fail without creation or clobber.
Matching partial uploads can retry from the original run after checksum checks.
Higher stable drafts/releases veto latest; abandoned newer drafts can delay it
without blocking immutable version publication. Keep snapshots/attestation
bundles and follow the shared recovery guide rather than rebuilding a version.

Wheel support is maintained in `[tool.cibuildwheel]`; platform rows live in
`[[tool.release-ci.targets]]` in `pyproject.toml`. Update the single `wheel-build`
cibuildwheel dependency pin and `uv.lock` together. Planning and builds use that
locked tool and configuration; publication rejects missing platform coverage.
