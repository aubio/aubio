/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

/** \mainpage 
 *
 * \section whatis Introduction
 *
 *  aubio is a library for audio labelling: it provides functions for pitch
 *  estimation, onset detection, beat tracking, and other annotation tasks.
 *
 *  \verbinclude README
 *
 * \section bugs bugs and todo
 *
 *  This software is under development. It needs debugging and
 *  optimisations.
 *
 *  See <a href='bug.html'>bugs</a> and <a href='todo.html'>todo</a> lists.
 * 
 */

#ifndef AUBIO_H
#define AUBIO_H

/**
 * Global aubio include file.
 * Programmers just need to include this file as:
 *
 * @code
 *   #include <aubio/aubio.h>
 * @endcode
 *
 * @file aubio.h
 */

#ifdef __cplusplus
extern "C"
{
#endif

/* first the generated config file */
#include "config.h"

/* in this order */
#include "types.h"
#include "fvec.h"
#include "cvec.h"
#include "lvec.h"
#include "musicutils.h"
#include "temporal/resampler.h"
#include "temporal/filter.h"
#include "temporal/biquad.h"
#include "temporal/a_weighting.h"
#include "temporal/c_weighting.h"
#include "spectral/fft.h"
#include "spectral/phasevoc.h"
#include "spectral/mfcc.h"
#include "pitch/pitch.h"
#include "onset/onset.h"
#include "tempo/tempo.h"

#if AUBIO_UNSTABLE
#include "vecutils.h"
#include "mathutils.h"
#include "utils/scale.h"
#include "utils/hist.h"
#include "spectral/tss.h"
#include "spectral/filterbank.h"
#include "spectral/filterbank_mel.h"
#include "pitch/pitchmcomb.h"
#include "pitch/pitchyin.h"
#include "pitch/pitchyinfft.h"
#include "pitch/pitchschmitt.h"
#include "pitch/pitchfcomb.h"
#include "onset/onsetdetection.h"
#include "spectral/spectral_centroid.h"
#include "onset/peakpick.h"
#include "tempo/beattracking.h"
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
