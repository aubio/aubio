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

/** @file
 * Private include file
 * 
 * This file is for inclusion from _within_ the library only.
 */

#ifndef _AUBIO_PRIV_H
#define _AUBIO_PRIV_H

/*********************
 *
 * External includes 
 *
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_STDLIB_H
#include <stdlib.h>
#endif

#if HAVE_STDIO_H
#include <stdio.h>
#endif

/* must be included before fftw3.h */
#if HAVE_COMPLEX_H
#include <complex.h>
#endif

#if HAVE_FFTW3_H
#include <fftw3.h>
#endif

#if HAVE_MATH_H
#include <math.h>
#endif

#if HAVE_STRINGS_H
#include <string.h>
#endif

#include "types.h"

/****
 * 
 * SYSTEM INTERFACE
 *
 */

/* Memory management */
#define AUBIO_MALLOC(_n)		malloc(_n)
#define AUBIO_REALLOC(_p,_n)		realloc(_p,_n)
#define AUBIO_NEW(_t)			(_t*)malloc(sizeof(_t))
#define AUBIO_ARRAY(_t,_n)		(_t*)malloc((_n)*sizeof(_t))
#define AUBIO_MEMCPY(_dst,_src,_n)	memcpy(_dst,_src,_n)
#define AUBIO_MEMSET(_dst,_src,_t)	memset(_dst,_src,_t)
#define AUBIO_FREE(_p)			free(_p)	


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
  AUBIO_FAIL = -1
} aubio_status;

#ifdef HAVE_C99_VARARGS_MACROS
#define AUBIO_ERR(...)               fprintf(stderr,__VA_ARGS__)
#define AUBIO_MSG(...)               fprintf(stdout,__VA_ARGS__)
#define AUBIO_DBG(...)               fprintf(stderr,__VA_ARGS__)
#else
#define AUBIO_ERR(format, args...)   fprintf(stderr, format , ##args)
#define AUBIO_MSG(format, args...)   fprintf(stdout, format , ##args)
#define AUBIO_DBG(format, args...)   fprintf(stderr, format , ##args)
#endif

#define AUBIO_QUIT(_s)               exit(_s)
#define AUBIO_SPRINTF                sprintf

/* Libc shortcuts */
#define PI         (M_PI)
#define TWO_PI     (PI*2.)

/* aliases to math.h functions */
#define EXP        expf
#define COS        cosf
#define SIN        sinf
#define ABS        fabsf
#define POW        powf
#define SQRT       sqrtf
#define LOG10      log10f
#define LOG        logf
#define FLOOR      floorf
#define TRUNC      truncf

/* aliases to complex.h functions */
#if !defined(HAVE_COMPLEX_H) || defined(WIN32)
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
#define DB2LIN(g) (POW(10.0f,(g)*0.05f))
#define LIN2DB(v) (20.0f*LOG10(v))
#define SQR(_a)   (_a*_a)

#define ELEM_SWAP(a,b) { register smpl_t t=(a);(a)=(b);(b)=t; }

#define UNUSED __attribute__((unused))

#endif/*_AUBIO_PRIV_H*/
