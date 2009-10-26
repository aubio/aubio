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

/** \file
 
  Onset detection functions
 
  All of the following onset detection function take as arguments the FFT of a
  windowed signal (as created with aubio_pvoc). They output one smpl_t per
  buffer and per channel (stored in a vector of size [channels]x[1]).
 
  These functions were first adapted from Juan Pablo Bello's code, and now
  include further improvements and modifications made within aubio.

*/


#ifndef ONSETDETECTION_H
#define ONSETDETECTION_H

#ifdef __cplusplus
extern "C" {
#endif

/** onsetdetection structure */
typedef struct _aubio_onsetdetection_t aubio_onsetdetection_t;
/** execute onset detection function on a spectral frame 

  Generic function to compute onset detection.
 
  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input signal spectrum as computed by aubio_pvoc_do
  \param onset output vector (one sample long, to send to the peak picking)

*/
void aubio_onsetdetection_do (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** creation of an onset detection object 

  \param type onset detection mode
  \param size length of the input spectrum frame
  \param channels number of input channels

*/
aubio_onsetdetection_t * new_aubio_onsetdetection(char_t * onset_mode, uint_t buf_size, uint_t channels);
/** deletion of an onset detection object

  \param o onset detection object as returned by new_aubio_onsetdetection()

*/
void del_aubio_onsetdetection(aubio_onsetdetection_t *o);

#ifdef __cplusplus
}
#endif

#endif /* ONSETDETECTION_H */
