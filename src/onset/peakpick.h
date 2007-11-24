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
 * peak picking utilities function
 * 
 * \todo check/fix peak picking
 */

#ifndef PEAKPICK_H
#define PEAKPICK_H

#ifdef __cplusplus
extern "C" {
#endif

/** function pointer to thresholding function */
typedef smpl_t (*aubio_thresholdfn_t)(fvec_t *input);
/** function pointer to peak-picking function */
typedef uint_t (*aubio_pickerfn_t)(fvec_t *input, uint_t pos);
/** peak-picker structure */
typedef struct _aubio_pickpeak_t aubio_pickpeak_t;

/** peak-picker creation function */
aubio_pickpeak_t * new_aubio_peakpicker(smpl_t threshold);
/** real time peak picking function */
uint_t aubio_peakpick_pimrt(fvec_t * DF, aubio_pickpeak_t * p);
/** function added by Miguel Ramirez to return the onset detection amplitude in peakval */
uint_t aubio_peakpick_pimrt_wt( fvec_t* DF, aubio_pickpeak_t* p, smpl_t* peakval );
/** get current peak value */
smpl_t aubio_peakpick_pimrt_getval(aubio_pickpeak_t * p);
/** destroy peak picker structure */
void del_aubio_peakpicker(aubio_pickpeak_t * p);

/** set peak picking threshold */
void aubio_peakpicker_set_threshold(aubio_pickpeak_t * p, smpl_t threshold);
/** get peak picking threshold */
smpl_t aubio_peakpicker_get_threshold(aubio_pickpeak_t * p);
/** set peak picker thresholding function */
void aubio_peakpicker_set_thresholdfn(aubio_pickpeak_t * p, aubio_thresholdfn_t thresholdfn);
/** get peak picker thresholding function */
aubio_thresholdfn_t aubio_peakpicker_get_thresholdfn(aubio_pickpeak_t * p);

#ifdef __cplusplus
}
#endif

#endif /* PEAKPICK_H */
