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
 *  various math functions
 *
 *  \todo multichannel (each function should return -or set- an array sized to
 *  the number of channel in the input vector)
 *
 *  \todo appropriate switches depending on types.h content
 */

#ifndef MATHUTILS_H
#define MATHUTILS_H

/** Window types 
 * 
 * inspired from 
 *
 *  - dafx : http://profs.sci.univr.it/%7Edafx/Final-Papers/ps/Bernardini.ps.gz
 *  - freqtweak : http://freqtweak.sf.net/
 *  - extace : http://extace.sf.net/
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	aubio_win_rectangle,          
	aubio_win_hamming,
	aubio_win_hanning,
	aubio_win_hanningz,
	aubio_win_blackman,
	aubio_win_blackman_harris,
	aubio_win_gaussian,
	aubio_win_welch,
	aubio_win_parzen
} aubio_window_type;

/** create window */
void aubio_window(smpl_t *w, uint_t size, aubio_window_type wintype);

/** principal argument
 *
 * mod(phase+PI,-TWO_PI)+PI 
 */
smpl_t aubio_unwrap2pi (smpl_t phase);

/** calculates the mean of a vector
 *
 * \bug mono 
 */
smpl_t vec_mean(fvec_t *s);
/** returns the max of a vector
 *
 * \bug mono 
 */
smpl_t vec_max(fvec_t *s);
/** returns the min of a vector
 *
 * \bug mono 
 */
smpl_t vec_min(fvec_t *s);
/** returns the index of the min of a vector
 *
 * \bug mono 
 */
uint_t vec_min_elem(fvec_t *s);
/** returns the index of the max of a vector
 *
 * \bug mono 
 */
uint_t vec_max_elem(fvec_t *s);
/** implement 'fftshift' like function
 * 
 * a[0]...,a[n/2],a[n/2+1],...a[n]
 * 
 * becomes
 * 
 * a[n/2+1],...a[n],a[0]...,a[n/2]
 */
void vec_shift(fvec_t *s);
/** returns sum */
smpl_t vec_sum(fvec_t *s);
/** returns energy 
 *
 * \bug mono 
 */
smpl_t vec_local_energy(fvec_t * f);
/** returns High Frequency Energy Content
 *
 * \bug mono */
smpl_t vec_local_hfc(fvec_t * f);
/** return alpha norm.
 *
 * alpha=2 means normalise variance. 
 * alpha=1 means normalise abs value. 
 * as alpha goes large, tends to normalisation 
 * by max value.
 *
 * \bug should not use POW :(
 */
smpl_t vec_alpha_norm(fvec_t * DF, smpl_t alpha);
/**  dc(min) removal */
void vec_dc_removal(fvec_t * mag);
/**  alpha normalisation */
void vec_alpha_normalise(fvec_t * mag, uint_t alpha);
/** add a constant to all members of a vector */
void vec_add(fvec_t * mag, smpl_t threshold);

/** compute adaptive threshold of input vector */ 
void vec_adapt_thres(fvec_t * vec, fvec_t * tmp, 
    uint_t win_post, uint_t win_pre);
/**  adaptative thresholding 
 *
 * y=fn_thresh(fn,x,post,pre)
 * compute adaptive threshold at each time 
 *    fn : a function name or pointer, eg 'median'
 *    x:   signal vector
 *    post: window length, causal part
 *    pre: window length, anti-causal part
 * Returns:
 *    y:   signal the same length as x
 *
 * Formerly median_thresh, used compute median over a 
 * window of post+pre+1 samples, but now works with any
 * function that takes a vector or matrix and returns a
 * 'representative' value for each column, eg
 *    medians=fn_thresh(median,x,8,8)  
 *    minima=fn_thresh(min,x,8,8)  
 * see SPARMS for explanation of post and pre
 */
smpl_t vec_moving_thres(fvec_t * vec, fvec_t * tmp, 
    uint_t win_post, uint_t win_pre, uint_t win_pos);

/** returns the median of the vector 
 *
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *
 *  This code by Nicolas Devillard - 1998. Public domain,
 *  available at http://ndevilla.free.fr/median/median/
 */
smpl_t vec_median(fvec_t * input);

/** finds exact maximum position by quadratic interpolation*/
smpl_t vec_quadint(fvec_t * x,uint_t pos);

/** finds exact minimum position by quadratic interpolation*/
smpl_t vec_quadint_min(fvec_t * x,uint_t pos, uint_t span);

/** Quadratic interpolation using Lagrange polynomial.
 *
 * inspired from ``Comparison of interpolation algorithms in real-time sound
 * processing'', Vladimir Arnost, 
 * 
 * estimate = s0 + (pf/2.)*((pf-3.)*s0-2.*(pf-2.)*s1+(pf-1.)*s2);
 *    where 
 *    \param s0,s1,s2 are 3 known points on the curve,
 *    \param pf is the floating point index [0;2]
 */
smpl_t aubio_quadfrac(smpl_t s0, smpl_t s1, smpl_t s2, smpl_t pf);

/** returns 1 if X1 is a peak and positive */
uint_t vec_peakpick(fvec_t * input, uint_t pos);

/** convert frequency bin to midi value */
smpl_t aubio_bintomidi(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
/** convert midi value to frequency bin */
smpl_t aubio_miditobin(smpl_t midi, smpl_t samplerate, smpl_t fftsize);
/** convert frequency bin to frequency (Hz) */
smpl_t aubio_bintofreq(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
/** convert frequency (Hz) to frequency bin */
smpl_t aubio_freqtobin(smpl_t freq, smpl_t samplerate, smpl_t fftsize);
/** convert frequency (Hz) to midi value (0-128) */
smpl_t aubio_freqtomidi(smpl_t freq);
/** convert midi value (0-128) to frequency (Hz) */
smpl_t aubio_miditofreq(smpl_t midi);

/** check if current buffer level is under a given threshold */
uint_t aubio_silence_detection(fvec_t * ibuf, smpl_t threshold);
/** get the current buffer level */
smpl_t aubio_level_detection(fvec_t * ibuf, smpl_t threshold);
/** 
 * calculate normalised autocorrelation function
 */
void aubio_autocorr(fvec_t * input, fvec_t * output);
/**
 * clean up cached memory at the end of program
 *
 * use this function at the end of programs to purge all
 * cached memory. so far this function is only used to clean
 * fftw cache.
 */
void aubio_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif

