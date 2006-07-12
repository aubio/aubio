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

#include "aubio_priv.h"
#include "sample.h"
#include "mathutils.h"
#include "biquad.h"
#include "peakpick.h"

/* peak picking parameters, default values in brackets
 *
 *	   [<----post----|--pre-->]
 *	.................|.............
 *	time->           ^now
 */
struct _aubio_pickpeak_t {
	/** thresh: offset threshold [0.033 or 0.01] */
	smpl_t threshold; 	
	/** win_post: median filter window length (causal part) [8] */
	uint_t 	win_post; 			
	/** pre: median filter window (anti-causal part) [post-1] */
	uint_t 	win_pre; 				
	/** threshfn: name or handle of fn for computing adaptive threshold [median]  */
	aubio_thresholdfn_t thresholdfn;
	/** picker: name or handle of fn for picking event times [peakpick] */
	aubio_pickerfn_t pickerfn;

	/** biquad lowpass filter */
	aubio_biquad_t * biquad;
	/** original onsets */
	fvec_t * onset_keep;
	/** modified onsets */
	fvec_t * onset_proc;
	/** peak picked window [3] */
	fvec_t * onset_peek;
	/** scratch pad for biquad and median */
	fvec_t * scratch;

	/** \bug should be used to calculate filter coefficients */
	/* cutoff: low-pass filter cutoff [0.34, 1] */
	/* smpl_t cutoff; */

	/* not used anymore */
	/* time precision [512/44100  winlength/samplerate, fs/buffer_size */
	/* smpl_t tau; */
	/* alpha: normalisation exponent [9] */
	/* smpl_t alpha; */
};


/** modified version for real time, moving mean adaptive threshold this method
 * is slightly more permissive than the offline one, and yelds to an increase
 * of false positives. best  */
uint_t aubio_peakpick_pimrt(fvec_t * onset,  aubio_pickpeak_t * p) {
	fvec_t * onset_keep = (fvec_t *)p->onset_keep;
	fvec_t * onset_proc = (fvec_t *)p->onset_proc;
	fvec_t * onset_peek = (fvec_t *)p->onset_peek;
	fvec_t * scratch    = (fvec_t *)p->scratch;
	smpl_t mean = 0., median = 0.;
	uint_t length = p->win_post + p->win_pre + 1;
	uint_t i = 0, j;

	/* store onset in onset_keep */
	/* shift all elements but last, then write last */
	/* for (i=0;i<channels;i++) { */
	for (j=0;j<length-1;j++) {
		onset_keep->data[i][j] = onset_keep->data[i][j+1];
		onset_proc->data[i][j] = onset_keep->data[i][j];
	}
	onset_keep->data[i][length-1] = onset->data[i][0];
	onset_proc->data[i][length-1] = onset->data[i][0];
	/* } */

	/* filter onset_proc */
	/** \bug filtfilt calculated post+pre times, should be only once !? */
	aubio_biquad_do_filtfilt(p->biquad,onset_proc,scratch);

	/* calculate mean and median for onset_proc */
	/* for (i=0;i<onset_proc->channels;i++) { */
	mean = vec_mean(onset_proc);
	/* copy to scratch */
	for (j = 0; j < length; j++)
		scratch->data[i][j] = onset_proc->data[i][j];
	median = p->thresholdfn(scratch);
	/* } */

	/* for (i=0;i<onset->channels;i++) { */
	/* shift peek array */
	for (j=0;j<3-1;j++) 
		onset_peek->data[i][j] = onset_peek->data[i][j+1];
	/* calculate new peek value */
	onset_peek->data[i][2] = 
		onset_proc->data[i][p->win_post] - median - mean * p->threshold;
	/* } */
	//AUBIO_DBG("%f\n", onset_peek->data[0][2]);
	return (p->pickerfn)(onset_peek,1);
}

/** this method returns the current value in the pick peaking buffer
 * after smoothing
 */
smpl_t aubio_peakpick_pimrt_getval(aubio_pickpeak_t * p) 
{
	uint_t i = 0;
	return p->onset_peek->data[i][1];
}

/** function added by Miguel Ramirez to return the onset detection amplitude in peakval */
uint_t aubio_peakpick_pimrt_wt(fvec_t * onset,  aubio_pickpeak_t * p, smpl_t* peakval) 
{
	uint_t isonset = 0;
	isonset = aubio_peakpick_pimrt(onset,p);

	//if ( isonset && peakval != NULL )
	if ( peakval != NULL )
		*peakval = aubio_peakpick_pimrt_getval(p); 

	return isonset;
}

void aubio_peakpicker_set_threshold(aubio_pickpeak_t * p, smpl_t threshold) {
	p->threshold = threshold;
	return;
}

smpl_t aubio_peakpicker_get_threshold(aubio_pickpeak_t * p) {
	return p->threshold;
}

void aubio_peakpicker_set_thresholdfn(aubio_pickpeak_t * p, aubio_thresholdfn_t thresholdfn) {
	p->thresholdfn = thresholdfn;
	return;
}

aubio_thresholdfn_t aubio_peakpicker_get_thresholdfn(aubio_pickpeak_t * p) {
	return (aubio_thresholdfn_t) (p->thresholdfn);
}

aubio_pickpeak_t * new_aubio_peakpicker(smpl_t threshold) {
	aubio_pickpeak_t * t = AUBIO_NEW(aubio_pickpeak_t);
	t->threshold = 0.1; /* 0.0668; 0.33; 0.082; 0.033; */
	if (threshold > 0. && threshold < 10.)
		t->threshold = threshold;
	t->win_post  = 5;
	t->win_pre   = 1;

	t->thresholdfn = (aubio_thresholdfn_t)(vec_median); /* (vec_mean); */
	t->pickerfn = (aubio_pickerfn_t)(vec_peakpick);

	t->scratch = new_fvec(t->win_post+t->win_pre+1,1);
	t->onset_keep = new_fvec(t->win_post+t->win_pre+1,1);
	t->onset_proc = new_fvec(t->win_post+t->win_pre+1,1);
	t->onset_peek = new_fvec(3,1);

	/* cutoff: low-pass filter cutoff [0.34, 1] */
	/* t->cutoff=0.34; */
	t->biquad = new_aubio_biquad(0.1600,0.3200,0.1600,-0.5949,0.2348);
	return t;
}

void del_aubio_peakpicker(aubio_pickpeak_t * p) {
	del_aubio_biquad(p->biquad);
	del_fvec(p->onset_keep);
	del_fvec(p->onset_proc);
	del_fvec(p->onset_peek);
	del_fvec(p->scratch);
	AUBIO_FREE(p);
}
