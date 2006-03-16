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

#ifndef _PITCHFCOMB_H
#define _PITCHFCOMB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_pitchfcomb_t aubio_pitchfcomb_t;

smpl_t aubio_pitchfcomb_detect (aubio_pitchfcomb_t *p, fvec_t * input);
aubio_pitchfcomb_t * new_aubio_pitchfcomb (uint_t bufsize, uint_t hopsize, uint_t samplerate);
void del_aubio_pitchfcomb (aubio_pitchfcomb_t *p);


#ifdef __cplusplus
}
#endif

#endif /* _PITCHFCOMB_H */


