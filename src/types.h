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
 
  \todo replace all known types with their alias (in midi*.[ch])
  
  \todo add unknown types aliases (char, FILE)
 
  \todo add OS switches
 
  \todo add long/float switches

*/

#ifdef __cplusplus
extern "C" {
#endif

/** short sample format (32 or 64 bits) */
typedef float        smpl_t;
//typedef double       smpl_t;
/** long sample format (64 bits or more) */
typedef double       lsmp_t;
//typedef long        lsmp_t;
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
