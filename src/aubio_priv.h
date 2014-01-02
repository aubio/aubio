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

/** @file
 * Private include file
 * 
 * This file is for inclusion from _within_ the library only.
 */

#ifndef _AUBIO__PRIV_H
#define _AUBIO__PRIV_H

/*********************
 *
 * External includes 
 *
 */

#include "config.h"

#if HAVE_STDLIB_H
#include <stdlib.h>
#endif

#if HAVE_STDIO_H
#include <stdio.h>
#endif

/* must be included before fftw3.h */
#ifdef HAVE_COMPLEX_H
#include <complex.h>
#endif

#if defined(HAVE_FFTW3) || defined(HAVE_FFTW3F)
#include <fftw3.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_LIMITS_H
#include <limits.h> // for CHAR_BIT, in C99 standard
#endif

#include "types.h"

#define AUBIO_UNSTABLE 1

#include "mathutils.h"

/****
 * 
 * SYSTEM INTERFACE
 *
 */

/* Memory management */
#define AUBIO_MALLOC(_n)             malloc(_n)
#define AUBIO_REALLOC(_p,_n)         realloc(_p,_n)
#define AUBIO_NEW(_t)                (_t*)calloc(sizeof(_t), 1)
#define AUBIO_ARRAY(_t,_n)           (_t*)calloc((_n)*sizeof(_t), 1)
#define AUBIO_MEMCPY(_dst,_src,_n)   memcpy(_dst,_src,_n)
#define AUBIO_MEMSET(_dst,_src,_t)   memset(_dst,_src,_t)
#define AUBIO_FREE(_p)               free(_p)


/* file interface */
#define AUBIO_FOPEN(_f,_m)           fopen(_f,_m)
#define AUBIO_FCLOSE(_f)             fclose(_f)
#define AUBIO_FREAD(_p,_s,_n,_f)     fread(_p,_s,_n,_f)
#define AUBIO_FSEEK(_f,_n,_set)      fseek(_f,_n,_set)

/* strings */
#define AUBIO_STRLEN(_s)             strlen(_s)
#define AUBIO_STRCMP(_s,_t)          strcmp(_s,_t)
#define AUBIO_STRNCMP(_s,_t,_n)      strncmp(_s,_t,_n)
#define AUBIO_STRCPY(_dst,_src)      strcpy(_dst,_src)
#define AUBIO_STRCHR(_s,_c)          strchr(_s,_c)
#ifdef strdup
#define AUBIO_STRDUP(s)              strdup(s)
#else
#define AUBIO_STRDUP(s)              AUBIO_STRCPY(AUBIO_MALLOC(AUBIO_STRLEN(s) + 1), s)
#endif


/* Error reporting */
typedef enum {
  AUBIO_OK = 0,
  AUBIO_FAIL = 1
} aubio_status;

#ifdef HAVE_C99_VARARGS_MACROS
#define AUBIO_ERR(...)               fprintf(stderr, "AUBIO ERROR: " __VA_ARGS__)
#define AUBIO_MSG(...)               fprintf(stdout, __VA_ARGS__)
#define AUBIO_DBG(...)               fprintf(stderr, __VA_ARGS__)
#define AUBIO_WRN(...)               fprintf(stderr, "AUBIO WARNING: " __VA_ARGS__)
#else
#define AUBIO_ERR(format, args...)   fprintf(stderr, "AUBIO ERROR: " format , ##args)
#define AUBIO_MSG(format, args...)   fprintf(stdout, format , ##args)
#define AUBIO_DBG(format, args...)   fprintf(stderr, format , ##args)
#define AUBIO_WRN(format, args...)   fprintf(stderr, "AUBIO WARNING: " format, ##args)
#endif

#define AUBIO_ERROR   AUBIO_ERR

#define AUBIO_QUIT(_s)               exit(_s)
#define AUBIO_SPRINTF                sprintf

/* Libc shortcuts */
#define PI         (M_PI)
#define TWO_PI     (PI*2.)

/* aliases to math.h functions */
#if !HAVE_AUBIO_DOUBLE
#define EXP        expf
#define COS        cosf
#define SIN        sinf
#define ABS        fabsf
#define POW        powf
#define SQRT       sqrtf
#define LOG10      log10f
#define LOG        logf
#define FLOOR      floorf
#define CEIL       ceilf
#define ATAN2      atan2f
#else
#define EXP        exp
#define COS        cos
#define SIN        sin
#define ABS        fabs
#define POW        pow
#define SQRT       sqrt
#define LOG10      log10
#define LOG        log
#define FLOOR      floor
#define CEIL       ceil
#define ATAN2      atan2
#endif
#define ROUND(x)   FLOOR(x+.5)

/* aliases to complex.h functions */
#if HAVE_AUBIO_DOUBLE || !defined(HAVE_COMPLEX_H) || defined(WIN32)
/* mingw32 does not know about c*f functions */
#define EXPC      cexp
/** complex = CEXPC(complex) */
#define CEXPC     cexp
/** sample = ARGC(complex) */
#define ARGC      carg
/** sample = ABSC(complex) norm */
#define ABSC      cabs
/** sample = REAL(complex) */
#define REAL      creal
/** sample = IMAG(complex) */
#define IMAG      cimag
#else
/** sample = EXPC(complex) */
#define EXPC      cexpf
/** complex = CEXPC(complex) */
#define CEXPC     cexp
/** sample = ARGC(complex) */
#define ARGC      cargf
/** sample = ABSC(complex) norm */
#define ABSC      cabsf
/** sample = REAL(complex) */
#define REAL      crealf
/** sample = IMAG(complex) */
#define IMAG      cimagf
#endif

/* handy shortcuts */
#define DB2LIN(g) (POW(10.0,(g)*0.05f))
#define LIN2DB(v) (20.0*LOG10(v))
#define SQR(_a)   (_a*_a)

#define MAX(a,b)  ( a > b ? a : b)
#define MIN(a,b)  ( a < b ? a : b)

#define ELEM_SWAP(a,b) { register smpl_t t=(a);(a)=(b);(b)=t; }

#define VERY_SMALL_NUMBER 2.e-42 //1.e-37

/** if ABS(f) < VERY_SMALL_NUMBER, returns 1, else 0 */
#define IS_DENORMAL(f) ABS(f) < VERY_SMALL_NUMBER

/** if ABS(f) < VERY_SMALL_NUMBER, returns 0., else f */
#define KILL_DENORMAL(f)  IS_DENORMAL(f) ? 0. : f

/** if f > VERY_SMALL_NUMBER, returns f, else returns VERY_SMALL_NUMBER */
#define CEIL_DENORMAL(f)  f < VERY_SMALL_NUMBER ? VERY_SMALL_NUMBER : f

#define SAFE_LOG10(f) LOG10(CEIL_DENORMAL(f))
#define SAFE_LOG(f)   LOG(CEIL_DENORMAL(f))

#define UNUSED __attribute__((unused))

#endif /* _AUBIO__PRIV_H */
