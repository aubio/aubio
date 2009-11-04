/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "spectral/fft.h"
#include "spectral/specdesc.h"
#include "mathutils.h"
#include "utils/hist.h"

void aubio_specdesc_energy(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_hfc(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_complex(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_phase(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_specdiff(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_kl(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_mkl(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_specdesc_specflux(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset);

extern void aubio_specdesc_centroid (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_spread (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_skewness (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_kurtosis (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_slope (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_decrease (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);
extern void aubio_specdesc_rolloff (aubio_specdesc_t * o, cvec_t * spec,
    fvec_t * desc);

/** onsetdetection types */
typedef enum {
        aubio_onset_energy,         /**< energy based */          
        aubio_onset_specdiff,       /**< spectral diff */         
        aubio_onset_hfc,            /**< high frequency content */
        aubio_onset_complex,        /**< complex domain */        
        aubio_onset_phase,          /**< phase fast */            
        aubio_onset_kl,             /**< Kullback Liebler */
        aubio_onset_mkl,            /**< modified Kullback Liebler */
        aubio_onset_specflux,       /**< spectral flux */
        aubio_specmethod_centroid,  /**< spectral centroid */
        aubio_specmethod_spread,    /**< spectral spread */
        aubio_specmethod_skewness,  /**< spectral skewness */
        aubio_specmethod_kurtosis,  /**< spectral kurtosis */
        aubio_specmethod_slope,     /**< spectral kurtosis */
        aubio_specmethod_decrease,  /**< spectral decrease */
        aubio_specmethod_rolloff,   /**< spectral rolloff */
        aubio_onset_default = aubio_onset_hfc, /**< default mode, set to hfc */
} aubio_specdesc_type;

/** structure to store object state */
struct _aubio_specdesc_t {
  aubio_specdesc_type onset_type; /**< onset detection type */
  /** Pointer to aubio_specdesc_<type> function */
  void (*funcpointer)(aubio_specdesc_t *o,
      cvec_t * fftgrain, fvec_t * onset);
  smpl_t threshold;      /**< minimum norm threshold for phase and specdiff */
  fvec_t *oldmag;        /**< previous norm vector */
  fvec_t *dev1 ;         /**< current onset detection measure vector */
  fvec_t *theta1;        /**< previous phase vector, one frame behind */
  fvec_t *theta2;        /**< previous phase vector, two frames behind */
  aubio_hist_t * histog; /**< histogram */
};


/* Energy based onset detection function */
void aubio_specdesc_energy  (aubio_specdesc_t *o UNUSED,
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
void aubio_specdesc_hfc(aubio_specdesc_t *o UNUSED,
    cvec_t * fftgrain, fvec_t * onset){
  uint_t i,j;
  for (i=0;i<fftgrain->channels;i++) {
    onset->data[i][0] = 0.;
    for (j=0;j<fftgrain->length;j++) {
      onset->data[i][0] += (j+1)*fftgrain->norm[i][j];
    }
  }
}


/* Complex Domain Method onset detection function */
void aubio_specdesc_complex (aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset) {
  uint_t i, j;
  uint_t nbins = fftgrain->length;
  for (i=0;i<fftgrain->channels; i++)  {
    onset->data[i][0] = 0.;
    for (j=0;j<nbins; j++)  {
      // compute the predicted phase
      o->dev1->data[i][j] = 2. * o->theta1->data[i][j] - o->theta2->data[i][j];
      // compute the euclidean distance in the complex domain
      // sqrt ( r_1^2 + r_2^2 - 2 * r_1 * r_2 * \cos ( \phi_1 - \phi_2 ) )
      onset->data[i][0] +=
        SQRT (ABS (SQR (o->oldmag->data[i][j]) + SQR (fftgrain->norm[i][j])
              - 2. * o->oldmag->data[i][j] * fftgrain->norm[i][j]
              * COS (o->dev1->data[i][j] - fftgrain->phas[i][j])));
      /* swap old phase data (need to remember 2 frames behind)*/
      o->theta2->data[i][j] = o->theta1->data[i][j];
      o->theta1->data[i][j] = fftgrain->phas[i][j];
      /* swap old magnitude data (1 frame is enough) */
      o->oldmag->data[i][j] = fftgrain->norm[i][j];
    }
  }
}


/* Phase Based Method onset detection function */
void aubio_specdesc_phase(aubio_specdesc_t *o, 
    cvec_t * fftgrain, fvec_t * onset){
  uint_t i, j;
  uint_t nbins = fftgrain->length;
  for (i=0;i<fftgrain->channels; i++)  {
    onset->data[i][0] = 0.0;
    o->dev1->data[i][0]=0.;
    for ( j=0;j<nbins; j++ )  {
      o->dev1->data[i][j] = 
        aubio_unwrap2pi(
            fftgrain->phas[i][j]
            -2.0*o->theta1->data[i][j]
            +o->theta2->data[i][j]);
      if ( o->threshold < fftgrain->norm[i][j] )
        o->dev1->data[i][j] = ABS(o->dev1->data[i][j]);
      else 
        o->dev1->data[i][j] = 0.0;
      /* keep a track of the past frames */
      o->theta2->data[i][j] = o->theta1->data[i][j];
      o->theta1->data[i][j] = fftgrain->phas[i][j];
    }
    /* apply o->histogram */
    aubio_hist_dyn_notnull(o->histog,o->dev1);
    /* weight it */
    aubio_hist_weight(o->histog);
    /* its mean is the result */
    onset->data[i][0] = aubio_hist_mean(o->histog);  
    //onset->data[i][0] = fvec_mean(o->dev1);
  }
}

/* Spectral difference method onset detection function */
void aubio_specdesc_specdiff(aubio_specdesc_t *o,
    cvec_t * fftgrain, fvec_t * onset){
  uint_t i, j;
  uint_t nbins = fftgrain->length;
  for (i=0;i<fftgrain->channels; i++)  {
    onset->data[i][0] = 0.0;
    for (j=0;j<nbins; j++)  {
      o->dev1->data[i][j] = SQRT(
          ABS(SQR( fftgrain->norm[i][j])
            - SQR(o->oldmag->data[i][j])));
      if (o->threshold < fftgrain->norm[i][j] )
        o->dev1->data[i][j] = ABS(o->dev1->data[i][j]);
      else 
        o->dev1->data[i][j] = 0.0;
      o->oldmag->data[i][j] = fftgrain->norm[i][j];
    }

    /* apply o->histogram (act somewhat as a low pass on the
     * overall function)*/
    aubio_hist_dyn_notnull(o->histog,o->dev1);
    /* weight it */
    aubio_hist_weight(o->histog);
    /* its mean is the result */
    onset->data[i][0] = aubio_hist_mean(o->histog);  

  }
}

/* Kullback Liebler onset detection function
 * note we use ln(1+Xn/(Xn-1+0.0001)) to avoid 
 * negative (1.+) and infinite values (+1.e-10) */
void aubio_specdesc_kl(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset){
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
void aubio_specdesc_mkl(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset){
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

/* Spectral flux */
void aubio_specdesc_specflux(aubio_specdesc_t *o, cvec_t * fftgrain, fvec_t * onset){ 
  uint_t i, j;
  for (i=0;i<fftgrain->channels;i++) {
    onset->data[i][0] = 0.;
    for (j=0;j<fftgrain->length;j++) {
      if (fftgrain->norm[i][j] > o->oldmag->data[i][j])
        onset->data[i][0] += fftgrain->norm[i][j] - o->oldmag->data[i][j];
      o->oldmag->data[i][j] = fftgrain->norm[i][j];
    }
  }
}

/* Generic function pointing to the choosen one */
void 
aubio_specdesc_do (aubio_specdesc_t *o, cvec_t * fftgrain, 
    fvec_t * onset) {
  o->funcpointer(o,fftgrain,onset);
}

/* Allocate memory for an onset detection 
 * depending on the choosen type, allocate memory as needed
 */
aubio_specdesc_t * 
new_aubio_specdesc (char_t * onset_mode, 
    uint_t size, uint_t channels){
  aubio_specdesc_t * o = AUBIO_NEW(aubio_specdesc_t);
  uint_t rsize = size/2+1;
  aubio_specdesc_type onset_type;
  if (strcmp (onset_mode, "energy") == 0)
      onset_type = aubio_onset_energy;
  else if (strcmp (onset_mode, "specdiff") == 0)
      onset_type = aubio_onset_specdiff;
  else if (strcmp (onset_mode, "hfc") == 0)
      onset_type = aubio_onset_hfc;
  else if (strcmp (onset_mode, "complexdomain") == 0)
      onset_type = aubio_onset_complex;
  else if (strcmp (onset_mode, "complex") == 0)
      onset_type = aubio_onset_complex;
  else if (strcmp (onset_mode, "phase") == 0)
      onset_type = aubio_onset_phase;
  else if (strcmp (onset_mode, "mkl") == 0)
      onset_type = aubio_onset_mkl;
  else if (strcmp (onset_mode, "kl") == 0)
      onset_type = aubio_onset_kl;
  else if (strcmp (onset_mode, "specflux") == 0)
      onset_type = aubio_onset_specflux;
  else if (strcmp (onset_mode, "centroid") == 0)
      onset_type = aubio_specmethod_centroid;
  else if (strcmp (onset_mode, "spread") == 0)
      onset_type = aubio_specmethod_spread;
  else if (strcmp (onset_mode, "skewness") == 0)
      onset_type = aubio_specmethod_skewness;
  else if (strcmp (onset_mode, "kurtosis") == 0)
      onset_type = aubio_specmethod_kurtosis;
  else if (strcmp (onset_mode, "slope") == 0)
      onset_type = aubio_specmethod_slope;
  else if (strcmp (onset_mode, "decrease") == 0)
      onset_type = aubio_specmethod_decrease;
  else if (strcmp (onset_mode, "rolloff") == 0)
      onset_type = aubio_specmethod_rolloff;
  else if (strcmp (onset_mode, "default") == 0)
      onset_type = aubio_onset_default;
  else {
      AUBIO_ERR("unknown spectral descriptor type %s.\n", onset_mode);
      onset_type = aubio_onset_default;
  }
  switch(onset_type) {
    /* for both energy and hfc, only fftgrain->norm is required */
    case aubio_onset_energy: 
      break;
    case aubio_onset_hfc:
      break;
      /* the other approaches will need some more memory spaces */
    case aubio_onset_complex:
      o->oldmag = new_fvec(rsize,channels);
      o->dev1   = new_fvec(rsize,channels);
      o->theta1 = new_fvec(rsize,channels);
      o->theta2 = new_fvec(rsize,channels);
      break;
    case aubio_onset_phase:
      o->dev1   = new_fvec(rsize,channels);
      o->theta1 = new_fvec(rsize,channels);
      o->theta2 = new_fvec(rsize,channels);
      o->histog = new_aubio_hist(0.0, PI, 10, channels);
      o->threshold = 0.1;
      break;
    case aubio_onset_specdiff:
      o->oldmag = new_fvec(rsize,channels);
      o->dev1   = new_fvec(rsize,channels);
      o->histog = new_aubio_hist(0.0, PI, 10, channels);
      o->threshold = 0.1;
      break;
    case aubio_onset_kl:
    case aubio_onset_mkl:
    case aubio_onset_specflux:
      o->oldmag = new_fvec(rsize,channels);
      break;
    default:
      break;
  }

  /* this switch could be in its own function to change between
   * detections on the fly. this would need getting rid of the switch
   * above and always allocate all the structure */

  switch(onset_type) {
    case aubio_onset_energy:
      o->funcpointer = aubio_specdesc_energy;
      break;
    case aubio_onset_hfc:
      o->funcpointer = aubio_specdesc_hfc;
      break;
    case aubio_onset_complex:
      o->funcpointer = aubio_specdesc_complex;
      break;
    case aubio_onset_phase:
      o->funcpointer = aubio_specdesc_phase;
      break;
    case aubio_onset_specdiff:
      o->funcpointer = aubio_specdesc_specdiff;
      break;
    case aubio_onset_kl:
      o->funcpointer = aubio_specdesc_kl;
      break;
    case aubio_onset_mkl:
      o->funcpointer = aubio_specdesc_mkl;
      break;
    case aubio_onset_specflux:
      o->funcpointer = aubio_specdesc_specflux;
      break;
    // for for the additional descriptors. these don't need additional memory
    case aubio_specmethod_centroid:
      o->funcpointer = aubio_specdesc_centroid;
      break;
    case aubio_specmethod_spread:
      o->funcpointer = aubio_specdesc_spread;
      break;
    case aubio_specmethod_skewness:
      o->funcpointer = aubio_specdesc_skewness;
      break;
    case aubio_specmethod_kurtosis:
      o->funcpointer = aubio_specdesc_kurtosis;
      break;
    case aubio_specmethod_slope:
      o->funcpointer = aubio_specdesc_slope;
      break;
    case aubio_specmethod_decrease:
      o->funcpointer = aubio_specdesc_decrease;
      break;
    case aubio_specmethod_rolloff:
      o->funcpointer = aubio_specdesc_rolloff;
      break;
    default:
      break;
  }
  o->onset_type = onset_type;
  return o;
}

void del_aubio_specdesc (aubio_specdesc_t *o){
  switch(o->onset_type) {
    /* for both energy and hfc, only fftgrain->norm is required */
    case aubio_onset_energy: 
      break;
    case aubio_onset_hfc:
      break;
      /* the other approaches will need some more memory spaces */
    case aubio_onset_complex:
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
    case aubio_onset_mkl:
    case aubio_onset_specflux:
      del_fvec(o->oldmag);
      break;
    default:
      break;
  }
  AUBIO_FREE(o);
}
