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

/* This algorithm was developped by A. de Cheveigne and H. Kawahara and
 * published in:
 * 
 * de Cheveigné, A., Kawahara, H. (2002) "YIN, a fundamental frequency
 * estimator for speech and music", J. Acoust. Soc. Am. 111, 1917-1930.  
 *
 * see http://recherche.ircam.fr/equipes/pcm/pub/people/cheveign.html
 */

#ifndef PITCHYINFFT_H
#define PITCHYINFFT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_pitchyinfft_t aubio_pitchyinfft_t;

smpl_t aubio_pitchyinfft_detect (aubio_pitchyinfft_t *p, fvec_t * input, smpl_t tol);
aubio_pitchyinfft_t * new_aubio_pitchyinfft (uint_t bufsize);
void del_aubio_pitchyinfft (aubio_pitchyinfft_t *p);

#ifdef __cplusplus
}
#endif

#endif /*PITCHYINFFT_H*/ 
