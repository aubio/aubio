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


/** structure to store object state */
struct _aubio_onsetdetection_t {
  aubio_onsetdetection_type type; /**< onset detection type */
  /** Pointer to aubio_onsetdetection_<type> function */
  void (*funcpointer)(aubio_onsetdetection_t *o,
    cvec_t * fftgrain, fvec_t * onset);
  smpl_t threshold;      /**< minimum norm threshold for phase and specdiff */
  fvec_t *oldmag;        /**< previous norm vector */
  fft_data_t *meas;      /**< current onset detection measure complex vector */
  fvec_t *dev1 ;         /**< current onset detection measure vector */
  fvec_t *theta1;        /**< previous phase vector, one frame behind */
  fvec_t *theta2;        /**< previous phase vector, two frames behind */
  aubio_hist_t * histog; /**< histogram */
};


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
void aubio_onsetdetection_complex (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset) {
	uint_t i, j;
	uint_t nbins = fftgrain->length;
	for (i=0;i<fftgrain->channels; i++)	{
		onset->data[i][0] = 0.;
		for (j=0;j<nbins; j++)	{
			o->dev1->data[i][j] 	 = aubio_unwrap2pi(
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
				aubio_unwrap2pi(
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

/* Kullback Liebler onset detection function
 * note we use ln(1+Xn/(Xn-1+0.0001)) to avoid 
 * negative (1.+) and infinite values (+1.e-10) */
void aubio_onsetdetection_kl(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset){
        uint_t i,j;
        for (i=0;i<fftgrain->channels;i++) {
                onset->data[i][0] = 0.;
                for (j=0;j<fftgrain->length;j++) {
                        onset->data[i][0] += fftgrain->norm[i][j]
                                *LOG(1.+fftgrain->norm[i][j]/(o->oldmag->data[i][j]+1.e-10));
                        o->oldmag->data[i][j] = fftgrain->norm[i][j];
                }
                if (isnan(onset->data[i][0])) onset->data[i][0] = 0.;
        }
}

/* Modified Kullback Liebler onset detection function
 * note we use ln(1+Xn/(Xn-1+0.0001)) to avoid 
 * negative (1.+) and infinite values (+1.e-10) */
void aubio_onsetdetection_mkl(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset){
	uint_t i,j;
	for (i=0;i<fftgrain->channels;i++) {
		onset->data[i][0] = 0.;
		for (j=0;j<fftgrain->length;j++) {
			onset->data[i][0] += LOG(1.+fftgrain->norm[i][j]/(o->oldmag->data[i][j]+1.e-10));
			o->oldmag->data[i][j] = fftgrain->norm[i][j];
		}
                if (isnan(onset->data[i][0])) onset->data[i][0] = 0.;
	}
}

/* Generic function pointing to the choosen one */
void 
aubio_onsetdetection(aubio_onsetdetection_t *o, cvec_t * fftgrain, 
		fvec_t * onset) {
	o->funcpointer(o,fftgrain,onset);
}

/* Allocate memory for an onset detection 
 * depending on the choosen type, allocate memory as needed
 */
aubio_onsetdetection_t * 
new_aubio_onsetdetection (aubio_onsetdetection_type type, 
		uint_t size, uint_t channels){
	aubio_onsetdetection_t * o = AUBIO_NEW(aubio_onsetdetection_t);
	uint_t rsize = size/2+1;
	switch(type) {
		/* for both energy and hfc, only fftgrain->norm is required */
		case aubio_onset_energy: 
			break;
		case aubio_onset_hfc:
			break;
		/* the other approaches will need some more memory spaces */
		case aubio_onset_complex:
			o->oldmag = new_fvec(rsize,channels);
			/** bug: must be complex array */
			o->meas = AUBIO_ARRAY(fft_data_t,size);
			o->dev1	 = new_fvec(rsize,channels);
			o->theta1 = new_fvec(rsize,channels);
			o->theta2 = new_fvec(rsize,channels);
			break;
		case aubio_onset_phase:
			o->dev1	 = new_fvec(rsize,channels);
			o->theta1 = new_fvec(rsize,channels);
			o->theta2 = new_fvec(rsize,channels);
			o->histog = new_aubio_hist(0.0f, PI, 10, channels);
			o->threshold = 0.1;
			break;
		case aubio_onset_specdiff:
			o->oldmag = new_fvec(rsize,channels);
			o->dev1	  = new_fvec(rsize,channels);
			o->histog = new_aubio_hist(0.0f, PI, 10, channels);
			o->threshold = 0.1;
			break;
                case aubio_onset_kl:
			o->oldmag = new_fvec(rsize,channels);
                        break;
                case aubio_onset_mkl:
			o->oldmag = new_fvec(rsize,channels);
                        break;
		default:
			break;
	}
	
	/* this switch could be in its own function to change between
	 * detections on the fly. this would need getting rid of the switch
	 * above and always allocate all the structure */

	switch(type) {
		case aubio_onset_energy:
			o->funcpointer = aubio_onsetdetection_energy;
			break;
		case aubio_onset_hfc:
			o->funcpointer = aubio_onsetdetection_hfc;
			break;
		case aubio_onset_complex:
			o->funcpointer = aubio_onsetdetection_complex;
			break;
		case aubio_onset_phase:
			o->funcpointer = aubio_onsetdetection_phase;
			break;
		case aubio_onset_specdiff:
			o->funcpointer = aubio_onsetdetection_specdiff;
			break;
                case aubio_onset_kl:
			o->funcpointer = aubio_onsetdetection_kl;
			break;
                case aubio_onset_mkl:
			o->funcpointer = aubio_onsetdetection_mkl;
			break;
		default:
			break;
	}
	o->type = type;
	return o;
}

void aubio_onsetdetection_free (aubio_onsetdetection_t *o){
  del_aubio_onsetdetection(o);
}

void del_aubio_onsetdetection (aubio_onsetdetection_t *o){

	switch(o->type) {
		/* for both energy and hfc, only fftgrain->norm is required */
		case aubio_onset_energy: 
			break;
		case aubio_onset_hfc:
			break;
		/* the other approaches will need some more memory spaces */
		case aubio_onset_complex:
	                AUBIO_FREE(o->meas);
                        del_fvec(o->oldmag);
	                del_fvec(o->dev1);
	                del_fvec(o->theta1);
                	del_fvec(o->theta2);
			break;
		case aubio_onset_phase:
	                del_fvec(o->dev1);
	                del_fvec(o->theta1);
                	del_fvec(o->theta2);
                        del_aubio_hist(o->histog);
			break;
		case aubio_onset_specdiff:
                        del_fvec(o->oldmag);
	                del_fvec(o->dev1);
                        del_aubio_hist(o->histog);
			break;
		case aubio_onset_kl:
	                del_fvec(o->oldmag);
			break;
		case aubio_onset_mkl:
	                del_fvec(o->oldmag);
			break;
		default:
			break;
	}
        AUBIO_FREE(o);
	
}
