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
 *
 * Onset detection functions
 *
 * These functions are adapted from Juan Pablo Bello matlab code.
 *
 * - all of the following onset detection function take as arguments the fft of
 *   a windowed signal ( be created with an aubio_pvoc).
 *
 *
 * (the phasevocoder implementation does implement an fftshift like)
 *
 * - they output one smpl_t per frame and per channel (stored in a fvec_t * of
 *   size [channels][1])
 *
 *  Some of the functions should be improved by - downsampling the input of the
 *  phasevocoder - oversampling the ouput 
 *
 *  \todo write a generic driver (with a phase vocoder and the appropriate 
 *  resampling)
 */


#ifndef ONSETDETECTION_H
#define ONSETDETECTION_H

#ifdef __cplusplus
extern "C" {
#endif

/** onsetdetection types */
typedef enum {
	energy,		/**< energy based */          
	specdiff,       /**< spectral diff */         
	hfc,		/**< high frequency content */
	complexdomain,  /**< complex domain */        
	phase		/**< phase fast */            
} aubio_onsetdetection_type;

/** onsetdetection structure */
typedef struct _aubio_onsetdetection_t aubio_onsetdetection_t;
/** Energy based onset detection function 
 *
 * calculates the local energy profile
 *
 * 	- buffer 1024
 * 	- overlap 512
 */
void aubio_onsetdetection_energy(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** High Frequency Content onset detection function
 *
 * 	- buffer 1024
 * 	- overlap 512
 */
void aubio_onsetdetection_hfc(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Complex Domain Method onset detection function 
 * 
 * 	From C. Duxbury & J. Pablo Bello
 * 		
 * 	- buffer 512
 * 	- overlap 128
 * 	- dowfact 8
 * 	- interpfact 2
 */
void aubio_onsetdetection_complex(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Phase Based Method onset detection function 
 *
 * 	- buffer 512
 * 	- overlap 128
 * 	- dowfact 8
 * 	- interpfact 2
 */
void aubio_onsetdetection_phase(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Spectral difference method onset detection function 
 *
 * 	- buffer 512
 * 	- overlap 128
 * 	- dowfact 8
 * 	- interpfact 2
 */
void aubio_onsetdetection_specdiff(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Generic function pointing to the choosen one */
void aubio_onsetdetection(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
/** Allocate memory for an onset detection */
aubio_onsetdetection_t * new_aubio_onsetdetection(aubio_onsetdetection_type type, uint_t size, uint_t channels);
/** Free memory for an onset detection */
void aubio_onsetdetection_free(aubio_onsetdetection_t *o);

#ifdef __cplusplus
}
#endif

#endif /* ONSETDETECTION_H */
