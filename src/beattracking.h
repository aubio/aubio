/*
         Copyright (C) 2003 Matthew Davies and Paul Brossier

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

#ifndef BEATTRACKING_H
#define BEATTRACKING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * beat tracking object
 */
typedef struct _aubio_beattracking_t aubio_beattracking_t;
/**
 * create beat tracking object
 * \param frame size [512] 
 * \param step increment - both in detection function samples -i.e. 11.6ms or 1 onset frame [128]
 * \param length over which beat period is found [128]
 * \param parameter for rayleigh weight vector - sets preferred tempo to 120bpm [43]
 * \param channel number (not functionnal) [1] */
aubio_beattracking_t * new_aubio_beattracking(uint_t winlen,
                uint_t channels);
/**
 * track the beat 
 * \param beat tracking object
 * \param current input detection function frame. already smoothed by adaptive median threshold. 
 * \param stored tactus candidate positions
 */
void aubio_beattracking_do(aubio_beattracking_t * bt, fvec_t * dfframes, fvec_t * out);
/**
 * delete beat tracker object
 */
void del_aubio_beattracking(aubio_beattracking_t * p);

#ifdef __cplusplus
}
#endif

#endif /* BEATTRACKING_H */
