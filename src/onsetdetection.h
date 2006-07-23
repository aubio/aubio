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

/** onsetdetection types */
typedef enum {
        aubio_onset_energy,         /**< energy based */          
        aubio_onset_specdiff,       /**< spectral diff */         
        aubio_onset_hfc,            /**< high frequency content */
        aubio_onset_complex,        /**< complex domain */        
        aubio_onset_phase,          /**< phase fast */            
        aubio_onset_kl,             /**< Kullback Liebler */
        aubio_onset_mkl             /**< modified Kullback Liebler */
} aubio_onsetdetection_type;

/** onsetdetection structure */
typedef struct _aubio_onsetdetection_t aubio_onsetdetection_t;
/** Energy based onset detection function 
 
  This function calculates the local energy of the input spectral frame.
  
  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_energy(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** High Frequency Content onset detection function
 
  This method computes the High Frequency Content (HFC) of the input spectral
  frame. The resulting function is efficient at detecting percussive onsets.

  Paul Masri. Computer modeling of Sound for Transformation and Synthesis of
  Musical Signal. PhD dissertation, University of Bristol, UK, 1996.
  
  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_hfc(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Complex Domain Method onset detection function 
 
  Christopher Duxbury, Mike E. Davies, and Mark B. Sandler. Complex domain
  onset detection for musical signals. In Proceedings of the Digital Audio
  Effects Conference, DAFx-03, pages 90-93, London, UK, 2003.

  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_complex(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Phase Based Method onset detection function 

  Juan-Pablo Bello, Mike P. Davies, and Mark B. Sandler. Phase-based note onset
  detection for music signals. In Proceedings of the IEEE International
  Conference on Acoustics Speech and Signal Processing, pages 441­444,
  Hong-Kong, 2003.

  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_phase(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Spectral difference method onset detection function 

  Jonhatan Foote and Shingo Uchihashi. The beat spectrum: a new approach to
  rhythm analysis. In IEEE International Conference on Multimedia and Expo
  (ICME 2001), pages 881­884, Tokyo, Japan, August 2001.

  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_specdiff(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Kullback-Liebler onset detection function 
  
  Stephen Hainsworth and Malcom Macleod. Onset detection in music audio
  signals. In Proceedings of the International Computer Music Conference
  (ICMC), Singapore, 2003.
  
  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_kl(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Modified Kullback-Liebler onset detection function 

  Paul Brossier, ``Automatic annotation of musical audio for interactive
  systems'', Chapter 2, Temporal segmentation, PhD thesis, Centre for Digital
  music, Queen Mary University of London, London, UK, 2006.

  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input spectral frame
  \param onset output onset detection function

*/
void aubio_onsetdetection_mkl(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** execute onset detection function on a spectral frame 

  Generic function to compute onset detection.
 
  \param o onset detection object as returned by new_aubio_onsetdetection()
  \param fftgrain input signal spectrum as computed by aubio_pvoc_do
  \param onset output vector (one sample long, to send to the peak picking)

*/
void aubio_onsetdetection(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** creation of an onset detection object 

  \param type onset detection mode
  \param size length of the input spectrum frame
  \param channels number of input channels

*/
aubio_onsetdetection_t * new_aubio_onsetdetection(aubio_onsetdetection_type type, uint_t size, uint_t channels);
/** deletion of an onset detection object

  \param o onset detection object as returned by new_aubio_onsetdetection()

*/
void del_aubio_onsetdetection(aubio_onsetdetection_t *o);
/** deletion of an onset detection object (obsolete)

  \param o onset detection object as returned by new_aubio_onsetdetection()

*/
void aubio_onsetdetection_free(aubio_onsetdetection_t *o);


#ifdef __cplusplus
}
#endif

#endif /* ONSETDETECTION_H */
