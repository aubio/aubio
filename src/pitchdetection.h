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

#ifndef PITCHAUTOTCORR_H
#define PITCHAUTOTCORR_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Generic method for pitch detection  

  This file creates the objects required for the computation of the selected
  pitch detection algorithm and output the results, in midi note or Hz.

*/

/** pitch detection algorithm */
typedef enum {
        aubio_pitch_yin,     /**< YIN algorithm */
        aubio_pitch_mcomb,   /**< Multi-comb filter */
        aubio_pitch_schmitt, /**< Schmitt trigger */
        aubio_pitch_fcomb,   /**< Fast comb filter */
        aubio_pitch_yinfft   /**< Spectral YIN */
} aubio_pitchdetection_type;

/** pitch detection output mode */
typedef enum {
        aubio_pitchm_freq,   /**< Frequency (Hz) */
        aubio_pitchm_midi,   /**< MIDI note (0.,127) */
        aubio_pitchm_cent,   /**< Cent */
        aubio_pitchm_bin     /**< Frequency bin (0,bufsize) */
} aubio_pitchdetection_mode;

/** pitch detection object */
typedef struct _aubio_pitchdetection_t aubio_pitchdetection_t;

/** execute pitch detection on an input signal frame
 
  \param p pitch detection object as returned by new_aubio_pitchdetection
  \param ibuf input signal of length hopsize 
 
*/
smpl_t aubio_pitchdetection(aubio_pitchdetection_t * p, fvec_t * ibuf);

/** change yin or yinfft tolerance threshold
  
  default is 0.15 for yin and 0.85 for yinfft
 
*/
void aubio_pitchdetection_set_yinthresh(aubio_pitchdetection_t *p, smpl_t thres);

/** deletion of the pitch detection object
 
  \param p pitch detection object as returned by new_aubio_pitchdetection
 
*/
void del_aubio_pitchdetection(aubio_pitchdetection_t * p);

/** creation of the pitch detection object
 
  \param bufsize size of the input buffer to analyse 
  \param hopsize step size between two consecutive analysis instant 
  \param channels number of channels to analyse
  \param samplerate sampling rate of the signal 
  \param type set pitch detection algorithm
  \param mode set pitch units for output
 
*/
aubio_pitchdetection_t * new_aubio_pitchdetection(uint_t bufsize, 
        uint_t hopsize, 
        uint_t channels,
        uint_t samplerate,
        aubio_pitchdetection_type type,
        aubio_pitchdetection_mode mode);

#ifdef __cplusplus
}
#endif

#endif /*PITCHDETECTION_H*/ 
