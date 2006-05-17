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

/** \file

  Beat tracking using a context dependant model

  This file implement the causal beat tracking algorithm designed by Matthew
  Davies and described in the following articles:

  Matthew E. P. Davies and Mark D. Plumbley. Causal tempo tracking of audio.
  In Proceedings of the International Symposium on Music Information Retrieval
  (ISMIR), pages 164­169, Barcelona, Spain, 2004.

  Matthew E. P. Davies, Paul Brossier, and Mark D. Plumbley. Beat tracking
  towards automatic musical accompaniment. In Proceedings of the Audio
  Engeeniring Society 118th Convention, Barcelona, Spain, May 2005.
  
*/
#ifndef BEATTRACKING_H
#define BEATTRACKING_H

#ifdef __cplusplus
extern "C" {
#endif

/** beat tracking object */
typedef struct _aubio_beattracking_t aubio_beattracking_t;

/** create beat tracking object

  \param winlen: frame size [512] 
  \param channels number (not functionnal) [1]

*/
aubio_beattracking_t * new_aubio_beattracking(uint_t winlen, uint_t channels);
/** track the beat 

  \param bt beat tracking object
  \param dfframes current input detection function frame, smoothed by
  adaptive median threshold. 
  \param out stored detected beat locations 

*/
void aubio_beattracking_do(aubio_beattracking_t * bt, fvec_t * dfframes, fvec_t * out);
/** delete beat tracking object

  \param p beat tracking object

*/
void del_aubio_beattracking(aubio_beattracking_t * p);

#ifdef __cplusplus
}
#endif

#endif /* BEATTRACKING_H */
