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
#include "fft.h"
#include "mathutils.h"
#include "hist.h"
#include "onsetdetection.h"


struct _aubio_onsetdetection_t {
	aubio_onsetdetection_type type;
	/** Pointer to aubio_onsetdetection_<type> function */
	void (*funcpointer)(aubio_onsetdetection_t *o,
			cvec_t * fftgrain, fvec_t * onset);
	smpl_t threshold;
	fvec_t *oldmag;
	fft_data_t *meas;
	fvec_t *dev1 ;
	fvec_t *theta1;
	fvec_t *theta2;
	aubio_hist_t * histog;
};


static aubio_onsetdetection_t * aubio_onsetdetection_alloc(aubio_onsetdetection_type type, uint_t size, uint_t channels);

/* Energy based onset detection function */
void aubio_onsetdetection_energy  (aubio_onsetdetection_t *o,
		cvec_t * fftgrain, fvec_t * onset) {
	uint_t i,j;
	for (i=0;i<fftgrain->channels;i++) {
		onset->data[i][0] = 0.;
		for (j=0;j<fftgrain->length;j++) {
			onset->data[i][0] += SQR(fftgrain->norm[i][j]);
		}
	}
}

/* High Frequency Content onset detection function */
void aubio_onsetdetection_hfc(aubio_onsetdetection_t *o,	cvec_t * fftgrain, fvec_t * onset){
	uint_t i,j;
	for (i=0;i<fftgrain->channels;i++) {
		onset->data[i][0] = 0.;
		for (j=0;j<fftgrain->length;j++) {
			onset->data[i][0] += (j+1)*fftgrain->norm[i][j];
		}
	}
}


/* Complex Domain Method onset detection function */
/* moved to /2 032402 */
void aubio_onsetdetection_complex (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset) {
	uint_t i, j;
	uint_t nbins = fftgrain->length;
	for (i=0;i<fftgrain->channels; i++)	{
		onset->data[i][0] = 0.;
		for (j=0;j<nbins; j++)	{
			o->dev1->data[i][j] 	 = unwrap2pi(
					fftgrain->phas[i][j]
					-2.0*o->theta1->data[i][j]+
					o->theta2->data[i][j]);
			o->meas[j] = fftgrain->norm[i][j]*CEXPC(I*o->dev1->data[i][j]);
			/* sum on all bins */
			onset->data[i][0]		 += //(fftgrain->norm[i][j]);
					SQRT(SQR( REAL(o->oldmag->data[i][j]-o->meas[j]) )
						+  SQR( IMAG(o->oldmag->data[i][j]-o->meas[j]) )
						);
			/* swap old phase data (need to remember 2 frames behind)*/
			o->theta2->data[i][j] = o->theta1->data[i][j];
			o->theta1->data[i][j] = fftgrain->phas[i][j];
			/* swap old magnitude data (1 frame is enough) */
			o->oldmag->data[i][j] = fftgrain->norm[i][j];
		}
	}
}


/* Phase Based Method onset detection function */
void aubio_onsetdetection_phase(aubio_onsetdetection_t *o, 
		cvec_t * fftgrain, fvec_t * onset){
	uint_t i, j;
	uint_t nbins = fftgrain->length;
	for (i=0;i<fftgrain->channels; i++)	{
		onset->data[i][0] = 0.0f;
		o->dev1->data[i][0]=0.;
		for ( j=0;j<nbins; j++ )	{
			o->dev1->data[i][j] = 
				unwrap2pi(
						fftgrain->phas[i][j]
						-2.0*o->theta1->data[i][j]
						+o->theta2->data[i][j]);
			if ( o->threshold < fftgrain->norm[i][j] )
				o->dev1->data[i][j] = ABS(o->dev1->data[i][j]);
			else 
				o->dev1->data[i][j] = 0.0f;
			/* keep a track of the past frames */
			o->theta2->data[i][j] = o->theta1->data[i][j];
			o->theta1->data[i][j] = fftgrain->phas[i][j];
		}
		/* apply o->histogram */
		aubio_hist_dyn_notnull(o->histog,o->dev1);
		/* weight it */
		aubio_hist_weigth(o->histog);
		/* its mean is the result */
		onset->data[i][0] = aubio_hist_mean(o->histog);	
		//onset->data[i][0] = vec_mean(o->dev1);
	}
}

/* Spectral difference method onset detection function */
/* moved to /2 032402 */
void aubio_onsetdetection_specdiff(aubio_onsetdetection_t *o,
		cvec_t * fftgrain, fvec_t * onset){
	uint_t i, j;
	uint_t nbins = fftgrain->length;
	for (i=0;i<fftgrain->channels; i++)	{
		onset->data[i][0] = 0.0f;
		for (j=0;j<nbins; j++)	{
			o->dev1->data[i][j] = SQRT(
					ABS(SQR( fftgrain->norm[i][j])
						- SQR(o->oldmag->data[i][j])));
			if (o->threshold < fftgrain->norm[i][j] )
				o->dev1->data[i][j] = ABS(o->dev1->data[i][j]);
			else 
				o->dev1->data[i][j] = 0.0f;
			o->oldmag->data[i][j] = fftgrain->norm[i][j];
		}

		/* apply o->histogram (act somewhat as a low pass on the
		 * overall function)*/
		aubio_hist_dyn_notnull(o->histog,o->dev1);
		/* weight it */
		aubio_hist_weigth(o->histog);
		/* its mean is the result */
		onset->data[i][0] = aubio_hist_mean(o->histog);	

	}
}

/* Generic function pointing to the choosen one */
void 
aubio_onsetdetection(aubio_onsetdetection_t *o, cvec_t * fftgrain, 
		fvec_t * onset) {
	o->funcpointer(o,fftgrain,onset);
}

/* Allocate memory for an onset detection */
aubio_onsetdetection_t * 
new_aubio_onsetdetection (aubio_onsetdetection_type type, 
		uint_t size, uint_t channels){
	return aubio_onsetdetection_alloc(type,size,channels);
}

/* depending on the choosen type, allocate memory as needed */
aubio_onsetdetection_t * 
aubio_onsetdetection_alloc (aubio_onsetdetection_type type, 
		uint_t size, uint_t channels){
	aubio_onsetdetection_t * o = AUBIO_NEW(aubio_onsetdetection_t);
	uint_t rsize = size/2+1;
	switch(type) {
		/* for both energy and hfc, only fftgrain->norm is required */
		case energy: 
			break;
		case hfc:
			break;
		/* the other approaches will need some more memory spaces */
		case complexdomain:
			o->oldmag = new_fvec(rsize,channels);
			/** bug: must be complex array */
			o->meas = AUBIO_ARRAY(fft_data_t,size);
			o->dev1	 = new_fvec(rsize,channels);
			o->theta1 = new_fvec(rsize,channels);
			o->theta2 = new_fvec(rsize,channels);
			break;
		case phase:
			o->dev1	 = new_fvec(rsize,channels);
			o->theta1 = new_fvec(rsize,channels);
			o->theta2 = new_fvec(rsize,channels);
			o->histog = new_aubio_hist(0.0f, PI, 10, channels);
			o->threshold = 0.1;
			break;
		case specdiff:
			o->oldmag = new_fvec(rsize,channels);
			o->dev1	  = new_fvec(rsize,channels);
			o->histog = new_aubio_hist(0.0f, PI, 10, channels);
			o->threshold = 0.1;
			break;
		default:
			break;
	}
	
	/* this switch could be in its own function to change between
	 * detections on the fly. this would need getting rid of the switch
	 * above and always allocate all the structure */

	switch(type) {
		case energy:
			o->funcpointer = aubio_onsetdetection_energy;
			break;
		case hfc:
			o->funcpointer = aubio_onsetdetection_hfc;
			break;
		case complexdomain:
			o->funcpointer = aubio_onsetdetection_complex;
			break;
		case phase:
			o->funcpointer = aubio_onsetdetection_phase;
			break;
		case specdiff:
			o->funcpointer = aubio_onsetdetection_specdiff;
			break;
		default:
			break;
	}
	o->type = type;
	return o;
}

void aubio_onsetdetection_free (aubio_onsetdetection_t *o){

	switch(o->type) {
		/* for both energy and hfc, only fftgrain->norm is required */
		case energy: 
			break;
		case hfc:
			break;
		/* the other approaches will need some more memory spaces */
		case complexdomain:
	                AUBIO_FREE(o->meas);
                        del_fvec(o->oldmag);
	                del_fvec(o->dev1);
	                del_fvec(o->theta1);
                	del_fvec(o->theta2);
			break;
		case phase:
	                del_fvec(o->dev1);
	                del_fvec(o->theta1);
                	del_fvec(o->theta2);
                        del_aubio_hist(o->histog);
			break;
		case specdiff:
                        del_fvec(o->oldmag);
	                del_fvec(o->dev1);
                        del_aubio_hist(o->histog);
			break;
		default:
			break;
	}
        AUBIO_FREE(o);
	
}
