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
 
  Spectral description functions
 
  All of the following spectral description functions take as arguments the FFT
  of a windowed signal (as created with aubio_pvoc). They output one smpl_t per
  buffer and per channel (stored in a vector of size [channels]x[1]).
 
  The following spectral description methods are available:

  \b \p energy : Energy based onset detection function 
 
  This function calculates the local energy of the input spectral frame.
  
  \b \p hfc : High Frequency Content onset detection function
 
  This method computes the High Frequency Content (HFC) of the input spectral
  frame. The resulting function is efficient at detecting percussive onsets.

  Paul Masri. Computer modeling of Sound for Transformation and Synthesis of
  Musical Signal. PhD dissertation, University of Bristol, UK, 1996.

  \b \p complex : Complex Domain Method onset detection function 
 
  Christopher Duxbury, Mike E. Davies, and Mark B. Sandler. Complex domain
  onset detection for musical signals. In Proceedings of the Digital Audio
  Effects Conference, DAFx-03, pages 90-93, London, UK, 2003.

  \b \p phase : Phase Based Method onset detection function 

  Juan-Pablo Bello, Mike P. Davies, and Mark B. Sandler. Phase-based note onset
  detection for music signals. In Proceedings of the IEEE International
  Conference on Acoustics Speech and Signal Processing, pages 441­444,
  Hong-Kong, 2003.

  \b \p specdiff : Spectral difference method onset detection function 

  Jonhatan Foote and Shingo Uchihashi. The beat spectrum: a new approach to
  rhythm analysis. In IEEE International Conference on Multimedia and Expo
  (ICME 2001), pages 881­884, Tokyo, Japan, August 2001.

  \b \p kl : Kullback-Liebler onset detection function 
  
  Stephen Hainsworth and Malcom Macleod. Onset detection in music audio
  signals. In Proceedings of the International Computer Music Conference
  (ICMC), Singapore, 2003.

  \b \p mkl : Modified Kullback-Liebler onset detection function 

  Paul Brossier, ``Automatic annotation of musical audio for interactive
  systems'', Chapter 2, Temporal segmentation, PhD thesis, Centre for Digital
  music, Queen Mary University of London, London, UK, 2006.

  \b \p specflux : Spectral Flux 

  Simon Dixon, Onset Detection Revisited, in ``Proceedings of the 9th
  International Conference on Digital Audio Effects'' (DAFx-06), Montreal,
  Canada, 2006. 
  
*/


#ifndef ONSETDETECTION_H
#define ONSETDETECTION_H

#ifdef __cplusplus
extern "C" {
#endif

/** spectral description structure */
typedef struct _aubio_specdesc_t aubio_specdesc_t;

/** execute spectral description function on a spectral frame 

  Generic function to compute spectral detescription.
 
  \param o spectral description object as returned by new_aubio_specdesc()
  \param fftgrain input signal spectrum as computed by aubio_pvoc_do
  \param desc output vector (one sample long, to send to the peak picking)

*/
void aubio_specdesc_do (aubio_specdesc_t * o, cvec_t * fftgrain,
    fvec_t * desc);

/** creation of a spectral description object 

  \param method spectral description method
  \param buf_size length of the input spectrum frame
  \param channels number of input channels

*/
aubio_specdesc_t *new_aubio_specdesc (char_t * method, uint_t buf_size,
    uint_t channels);

/** deletion of a spectral descriptor 

  \param o spectral descriptor object as returned by new_aubio_specdesc()

*/
void del_aubio_specdesc (aubio_specdesc_t * o);

#ifdef __cplusplus
}
#endif

#endif /* ONSETDETECTION_H */
