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

#if HAVE_COMPLEX_H
#include <complex.h>
#endif
/*
#include <complex.h>
#include <fftw3.h>
#define FFTW_TYPE fftwf_complex
*/
#if HAVE_FFTW3_H
#include <fftw3.h>
//#define FFTW_TYPE fftwf_complex
#endif

#if HAVE_MATH_H
#include <math.h>
#endif

#if HAVE_STRINGS_H
#include <string.h>
#endif

#ifdef ALSA_SUPPORT
#ifdef LADCCA_SUPPORT
#include <ladcca/ladcca.h>
extern cca_client_t * aubio_cca_client;
#endif /* LADCCA_SUPPORT */
#endif /* ALSA_SUPPORT */


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
#define AUBIO_MEMSET(_dst,_src,_t)	memset(_dst,_src,sizeof(_t))
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

#endif/*_AUBIO_PRIV_H*/
