/*
	 Copyright (C) 2003 Paul Brossier

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

#ifndef AUBIO_TYPES_H
#define AUBIO_TYPES_H

/** \file
 
  Definition of data types used in aubio
 
*/

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"

#ifndef HAVE_AUBIO_DOUBLE
#define HAVE_AUBIO_DOUBLE 0
#endif

#if HAVE_AUBIO_DOUBLE
#define AUBIO_SINGLE_PRECISION 0
#else
#define AUBIO_SINGLE_PRECISION 1
#endif

/** short sample format (32 or 64 bits) */
#if AUBIO_SINGLE_PRECISION
typedef float        smpl_t;
#define AUBIO_SMPL_FMT "%f"
#else
typedef double       smpl_t;
#define AUBIO_SMPL_FMT "%lf"
#endif
/** long sample format (64 bits or more) */
#if AUBIO_SINGLE_PRECISION 
typedef double       lsmp_t;
#define AUBIO_LSMP_FMT "%lf"
#else
typedef long double  lsmp_t;
#define AUBIO_LSMP_FMT "%Lf"
#endif
/** unsigned integer */
typedef unsigned int uint_t;
/** signed integer */
typedef int          sint_t;
/** files */
//typedef FILE         audio_file_t;

#ifdef __cplusplus
}
#endif

#endif/*AUBIO_TYPES_H*/
