# Changelog

## [0.4.12](https://github.com/LedFx/aubio-ledfx/compare/v0.4.11...v0.4.12) (2026-10-03)


### Bug Fixes

* copy file paths safely, rejecting paths too long instead of truncating them ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* design A/C-weighting for any samplerate, and fail cleanly instead of crashing in pitch mcomb ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* intentional float precision, and the bugs implicit conversions hid (pitch mcomb/schmitt, beat tracker prior, onset/tempo delays, 64-bit wav seeks, long sndfile durations) ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* **io:** source_apple_audio fails when its path is rejected ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* **python:** build the bindings in double precision when the library is ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* **python:** give every CPython slot its exact signature, and build warning-free ([de98bdd](https://github.com/LedFx/aubio-ledfx/commit/de98bdd516ffb83627117ebd292bd1a8b9bd2504))
* **tempo:** unbiased beat period, fair phase choice, window sized in time, latency taken back ([#63](https://github.com/LedFx/aubio-ledfx/issues/63)) ([595c719](https://github.com/LedFx/aubio-ledfx/commit/595c7197857e5e6c2100ed92edf6e27e6dbe4d39))
