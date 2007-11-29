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

  Pitch detection using multiple-comb filter

  This fundamental frequency estimation algorithm implements spectral
  flattening, multi-comb filtering and peak histogramming. 

  This method was designed by Juan P. Bello and described in:

  Juan-Pablo Bello. ``Towards the Automated Analysis of Simple Polyphonic
  Music''.  PhD thesis, Centre for Digital Music, Queen Mary University of
  London, London, UK, 2003.

*/

#ifndef PITCHMCOMB_H
#define PITCHMCOMB_H

#ifdef __cplusplus
extern "C" {
#endif

/** pitch detection object */
typedef struct _aubio_pitchmcomb_t aubio_pitchmcomb_t;

/** execute pitch detection on an input spectral frame
 
  \param p pitch detection object as returned by new_aubio_pitchmcomb
  \param fftgrain input signal spectrum as computed by aubio_pvoc_do 
 
*/
smpl_t aubio_pitchmcomb_detect(aubio_pitchmcomb_t * p, cvec_t * fftgrain);
/** select the best candidates */
uint_t aubio_pitch_cands(aubio_pitchmcomb_t * p, cvec_t * fftgrain, smpl_t * cands);
/** creation of the pitch detection object
 
  \param bufsize size of the input buffer to analyse 
  \param hopsize step size between two consecutive analysis instant 
  \param channels number of channels to analyse
  \param samplerate sampling rate of the signal 
 
*/
aubio_pitchmcomb_t * new_aubio_pitchmcomb(uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate);
/** deletion of the pitch detection object
 
  \param p pitch detection object as returned by new_aubio_pitchfcomb
 
*/
void del_aubio_pitchmcomb(aubio_pitchmcomb_t *p);

#ifdef __cplusplus
}
#endif

#endif/*PITCHMCOMB_H*/ 
