/*
   Copyright (C) 2006 Amaury Hazan
   Ported to aubio from LibXtract
   http://libxtract.sourceforge.net/
   

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

#ifndef BARK_H
#define BARK_H


// Initalization

/** \brief A function to initialise bark filter bounds
 * 
 * A pointer to an array of BARK_BANDS ints most be passed in, and is populated with BARK_BANDS fft bin numbers representing the limits of each band 
 *
 * \param N: the audio block size
 * \param sr: The sample audio sample rate
 * \param *band_limits: a pointer to an array of BARK_BANDS ints
 */
int xtract_init_bark(int N, float sr, int *band_limits);

// Computation

/** \brief Extract Bark band coefficients based on a method   
 * \param *data: a pointer to the first element in an array of floats representing the magnitude coefficients from the magnitude spectrum of an audio vector, (e.g. the first half of the array pointed to by *result from xtract_spectrum().
 * \param N: the number of array elements to be considered
 * \param *argv: a pointer to an array of ints representing the limits of each bark band. This can be obtained  by calling xtract_init_bark.
 * \param *result: a pointer to an array containing resultant bark coefficients
 *
 * The limits array pointed to by *argv must be obtained by first calling xtract_init_bark
 * 
 */
int xtract_bark_coefficients(const float *data, const int N, const void *argv, float *result);


#endif