/*
   Copyright (C) 2003 Paul Brossier <piem@altern.org>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
   
*/

/** \mainpage 
 *
 * \section whatis Introduction
 *
 *  Aubio is a library for audio labelling: it provides functions for pitch
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
 * Global Aubio include file.
 * Programmers just need to include this file as:
 *
 * @code
 *   #include <aubio/aubio.h>
 * @endcode
 *
 * @file aubio.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/* first the generated config file */
#include "config.h"
 
/* in this order */
#include "types.h"
#include "fvec.h"
#include "cvec.h"
#include "lvec.h"
#include "mathutils.h"
#include "utils/scale.h"
#include "utils/hist.h"
#include "spectral/tss.h"
#include "temporal/resample.h"
#include "temporal/biquad.h"
#include "temporal/filter.h"
#include "temporal/adesign.h"
#include "temporal/cdesign.h"
#include "spectral/filterbank.h"
#include "spectral/mfcc.h"
#include "spectral/fft.h"
#include "spectral/phasevoc.h"
#include "spectral/spectral_centroid.h"
#include "pitch/pitchdetection.h"
#include "pitch/pitchmcomb.h"
#include "pitch/pitchyin.h"
#include "pitch/pitchyinfft.h"
#include "pitch/pitchschmitt.h"
#include "pitch/pitchfcomb.h"
#include "onset/onsetdetection.h"
#include "onset/onset.h"
#include "onset/peakpick.h"
#include "tempo/beattracking.h"
#include "tempo/tempo.h"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif

