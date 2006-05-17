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
#include "pitchmcomb.h"

#define CAND_SWAP(a,b) { register aubio_spectralcandidate_t *t=(a);(a)=(b);(b)=t; }

typedef struct _aubio_spectralpeak_t aubio_spectralpeak_t;
typedef struct _aubio_spectralcandidate_t aubio_spectralcandidate_t;
uint_t aubio_pitchmcomb_get_root_peak(aubio_spectralpeak_t * peaks, uint_t length);
uint_t aubio_pitchmcomb_quadpick(aubio_spectralpeak_t * spectral_peaks, fvec_t * X);
void aubio_pitchmcomb_spectral_pp(aubio_pitchmcomb_t * p, fvec_t * oldmag);
void aubio_pitchmcomb_combdet(aubio_pitchmcomb_t * p, fvec_t * newmag);
/* not used but useful : sort by amplitudes (or anything else)
 * sort_pitchpeak(peaks, length);
 */
/** spectral_peak comparison function (must return signed int) */
static sint_t aubio_pitchmcomb_sort_peak_comp(const void *x, const void *y);
/** sort spectral_peak against their mag */
void aubio_pitchmcomb_sort_peak(aubio_spectralpeak_t * peaks, uint_t nbins);

/** sort spectral_candidate against their comb ene */
void aubio_pitchmcomb_sort_cand_ene(aubio_spectralcandidate_t ** candidates, uint_t nbins);
/** sort spectral_candidate against their frequency */
void aubio_pitchmcomb_sort_cand_freq(aubio_spectralcandidate_t ** candidates, uint_t nbins);

struct _aubio_pitchmcomb_t {
  smpl_t threshold;                        /**< offset threshold [0.033 or 0.01]     */
  smpl_t alpha;                            /**< normalisation exponent [9]           */
  smpl_t cutoff;                           /**< low-pass filter cutoff [0.34, 1]     */
  smpl_t tol;                              /**< tolerance [0.05]                     */
  smpl_t tau;                              /**< frequency precision [44100/4096]     */
  uint_t win_post;                         /**< median filter window length          */
  uint_t win_pre;                          /**< median filter window                 */
  uint_t ncand;                            /**< maximum number of candidates (combs) */
  uint_t npartials;                        /**< maximum number of partials per combs */
  uint_t count;                            /**< picked picks                         */
  uint_t goodcandidate;                    /**< best candidate                       */
  uint_t spec_partition;                   /**< spectrum partition to consider       */
  aubio_spectralpeak_t * peaks;            /**< up to length win/spec_partition      */
  aubio_spectralcandidate_t ** candidates; /** up to five candidates                 */
  /* some scratch pads */
  /** \bug  (unnecessary copied from fftgrain?) */
  fvec_t * newmag;                         /**< vec to store mag                     */
  fvec_t * scratch;                        /**< vec to store modified mag            */
  fvec_t * scratch2;                       /**< vec to compute moving median         */
  fvec_t * theta;                         /**< vec to store phase                     */
  smpl_t phasediff;
  smpl_t phasefreq;
  /** threshfn: name or handle of fn for computing adaptive threshold [median] */
  /** aubio_thresholdfn_t thresholdfn; */
  /** picker: name or handle of fn for picking event times [quadpick] */
  /** aubio_pickerfn_t pickerfn; */
};

/** spectral peak object */
struct _aubio_spectralpeak_t {
  uint_t bin;     /**< bin [0-(length-1)] */
  smpl_t ebin;    /**< estimated bin */
  smpl_t mag;     /**< peak magnitude */
};

/** spectral candidates array object */
struct _aubio_spectralcandidate_t {
  smpl_t ebin;    /**< interpolated bin */
  smpl_t * ecomb; /**< comb */
  smpl_t ene;     /**< candidate energy */
  smpl_t len;     /**< length */
};


smpl_t aubio_pitchmcomb_detect(aubio_pitchmcomb_t * p, cvec_t * fftgrain) {
  uint_t i=0,j;
  smpl_t instfreq;
  fvec_t * newmag = (fvec_t *)p->newmag;
  //smpl_t hfc; //fe=instfreq(theta1,theta,ops); //theta1=theta;
  /* copy incoming grain to newmag */
  for (j=0; j< newmag->length; j++)
    newmag->data[i][j]=fftgrain->norm[i][j];
  /* detect only if local energy > 10. */ 
  //if (vec_local_energy(newmag)>10.) {
    //hfc = vec_local_hfc(newmag); //not used
    aubio_pitchmcomb_spectral_pp(p, newmag);
    aubio_pitchmcomb_combdet(p,newmag);
    //aubio_pitchmcomb_sort_cand_freq(p->candidates,p->ncand);
    //return p->candidates[p->goodcandidate]->ebin;
  j = (uint_t)FLOOR(p->candidates[p->goodcandidate]->ebin+.5);
  instfreq  = aubio_unwrap2pi(fftgrain->phas[0][j]
		  - p->theta->data[0][j] - j*p->phasediff);
  instfreq *= p->phasefreq;
  /* store phase for next run */
  for (j=0; j< p->theta->length; j++) {
    p->theta->data[i][j]=fftgrain->phas[i][j];
  }
  //return p->candidates[p->goodcandidate]->ebin;
  return FLOOR(p->candidates[p->goodcandidate]->ebin+.5) + instfreq;
  /*} else {
    return -1.;
  }*/
}

uint_t aubio_pitch_cands(aubio_pitchmcomb_t * p, cvec_t * fftgrain, 
    smpl_t * cands) {
  uint_t i=0,j;
  uint_t k;
  fvec_t * newmag = (fvec_t *)p->newmag;
  aubio_spectralcandidate_t ** scands = 
    (aubio_spectralcandidate_t **)(p->candidates);
  //smpl_t hfc; //fe=instfreq(theta1,theta,ops); //theta1=theta;
  /* copy incoming grain to newmag */
  for (j=0; j< newmag->length; j++)
    newmag->data[i][j]=fftgrain->norm[i][j];
  /* detect only if local energy > 10. */ 
  if (vec_local_energy(newmag)>10.)	{
    /* hfc = vec_local_hfc(newmag); do not use */
    aubio_pitchmcomb_spectral_pp(p, newmag);
    aubio_pitchmcomb_combdet(p,newmag);
    aubio_pitchmcomb_sort_cand_freq(scands,p->ncand);
    /* store ncand comb energies in cands[1:ncand] */ 
    for (k = 0; k<p->ncand; k++) 
      cands[k] = p->candidates[k]->ene;
    /* store ncand[end] freq in cands[end] */ 
    cands[p->ncand] = p->candidates[p->ncand-1]->ebin;
    return 1;
  } else {
    for (k = 0; k<p->ncand; k++)
      cands[k] = 0;
    return 0;
  }
}

void aubio_pitchmcomb_spectral_pp(aubio_pitchmcomb_t * p, fvec_t * newmag) {
  fvec_t * mag = (fvec_t *)p->scratch;
  fvec_t * tmp = (fvec_t *)p->scratch2;
  uint_t i=0,j;
  uint_t length = mag->length;
  /* copy newmag to mag (scracth) */
  for (j=0;j<length;j++) {
    mag->data[i][j] = newmag->data[i][j]; 
  }
  vec_dc_removal(mag);               /* dc removal           */
  vec_alpha_normalise(mag,p->alpha); /* alpha normalisation  */
  /* skipped */                      /* low pass filtering   */
  /** \bug vec_moving_thres may write out of bounds */
  vec_adapt_thres(mag,tmp,p->win_post,p->win_pre); /* adaptative threshold */
  vec_add(mag,-p->threshold);        /* fixed threshold      */
  {
    aubio_spectralpeak_t * peaks = (aubio_spectralpeak_t *)p->peaks;
    uint_t count;
    /*  return bin and ebin */
    count = aubio_pitchmcomb_quadpick(peaks,mag);
    for (j=0;j<count;j++) 
      peaks[j].mag = newmag->data[i][peaks[j].bin];
    /* reset non peaks */
    for (j=count;j<length;j++)
      peaks[j].mag = 0.;
    p->peaks = peaks;
    p->count = count;
  }
}

void aubio_pitchmcomb_combdet(aubio_pitchmcomb_t * p, fvec_t * newmag) {
  aubio_spectralpeak_t * peaks = (aubio_spectralpeak_t *)p->peaks;
  aubio_spectralcandidate_t ** candidate = 
    (aubio_spectralcandidate_t **)p->candidates;

  /* parms */
  uint_t N = p->npartials; /* maximum number of partials to be considered 10 */
  uint_t M = p->ncand;  /* maximum number of combs to be considered 5 */
  uint_t length = newmag->length;
  uint_t count = p->count;
  uint_t k;
  uint_t l;
  uint_t d;
  uint_t curlen;

  smpl_t delta2;
  smpl_t xx;
  uint_t position = 0;

  uint_t root_peak = 0;
  uint_t tmpl = 0;
  smpl_t tmpene = 0.;

  /* get the biggest peak in the spectrum */
  root_peak = aubio_pitchmcomb_get_root_peak(peaks,count);
  /* not enough partials in highest notes, could be forced */
  //if (peaks[root_peak].ebin >= aubio_miditofreq(85.)/p->tau) N=2;
  //if (peaks[root_peak].ebin >= aubio_miditofreq(90.)/p->tau) N=1;
  /* now calculate the energy of each of the 5 combs */
  for (l=0;l<M;l++) {
    smpl_t scaler = (1./(l+1.));
    candidate[l]->ene = 0.; /* reset ene and len sums */
    candidate[l]->len = 0.;
    candidate[l]->ebin=scaler*peaks[root_peak].ebin;
    /* if less than N peaks available, curlen < N */
    curlen = (uint_t)FLOOR(length/(candidate[l]->ebin));
    curlen = (N < curlen )? N : curlen;
    /* fill candidate[l]->ecomb[k] with (k+1)*candidate[l]->ebin */
    for (k=0;k<curlen;k++)
      candidate[l]->ecomb[k]=(candidate[l]->ebin)*(k+1.);
    for (k=curlen;k<length;k++)
      candidate[l]->ecomb[k]=0.;
    /* for each in candidate[l]->ecomb[k] */
    for (k=0;k<curlen;k++) {
      xx = 100000.;
      /** get the candidate->ecomb the closer to peaks.ebin 
       * (to cope with the inharmonicity)*/
      for (d=0;d<count;d++) { 
        delta2 = ABS(candidate[l]->ecomb[k]-peaks[d].ebin);
        if (delta2 <= xx) {
          position = d;
          xx = delta2;
        }
      }
      /* for a Q factor of 17, maintaining "constant Q filtering", 
       * and sum energy and length over non null combs */
      if ( 17. * xx < candidate[l]->ecomb[k] ) {
        candidate[l]->ecomb[k]=peaks[position].ebin;
        candidate[l]->ene += /* ecomb rounded to nearest int */
          POW(newmag->data[0][(uint_t)FLOOR(candidate[l]->ecomb[k]+.5)],0.25);
        candidate[l]->len += 1./curlen;
      } else
        candidate[l]->ecomb[k]=0.;
    }
    /* punishment */
    /*if (candidate[l]->len<0.6)
      candidate[l]->ene=0.; */
    /* remember best candidate energy (in polyphonic, could check for
     * tmpene*1.1 < candidate->ene to reduce jumps towards low frequencies) */
    if (tmpene < candidate[l]->ene) {
      tmpl = l;
      tmpene = candidate[l]->ene;
    }
  }
  //p->candidates=candidate;
  //p->peaks=peaks;
  p->goodcandidate = tmpl;
}

/** T=quadpick(X): return indices of elements of X which are peaks and positive
 * exact peak positions are retrieved by quadratic interpolation
 *
 * \bug peak-picking too picky, sometimes counts too many peaks ? 
 */
uint_t aubio_pitchmcomb_quadpick(aubio_spectralpeak_t * spectral_peaks, fvec_t * X){
  uint_t i, j, ispeak, count = 0;
  for (i=0;i<X->channels;i++)
    for (j=1;j<X->length-1;j++)	{
      ispeak = vec_peakpick(X,j);
      if (ispeak) {
        count += ispeak;
        spectral_peaks[count-1].bin = j;
        spectral_peaks[count-1].ebin = vec_quadint(X,j) - 1.;
      }
    }
  return count;
}

/* get predominant partial */
uint_t aubio_pitchmcomb_get_root_peak(aubio_spectralpeak_t * peaks, uint_t length) {
  uint_t i,pos=0;
  smpl_t tmp = 0.;
  for (i=0;i<length;i++)
    if (tmp <= peaks[i].mag) {
      pos = i;
      tmp = peaks[i].mag;
    }
  return pos;
}

void aubio_pitchmcomb_sort_peak(aubio_spectralpeak_t * peaks, uint_t nbins) {
  qsort(peaks, nbins, sizeof(aubio_spectralpeak_t), 
      aubio_pitchmcomb_sort_peak_comp);
}
static sint_t aubio_pitchmcomb_sort_peak_comp(const void *x, const void *y) {
  return (((aubio_spectralpeak_t *)y)->mag - ((aubio_spectralpeak_t *)x)->mag);
}


void aubio_pitchmcomb_sort_cand_ene(aubio_spectralcandidate_t ** candidates, uint_t nbins) {
  uint_t cur = 0;
  uint_t run = 0;
  for (cur=0;cur<nbins;cur++) {
    run = cur + 1;
    for (run=cur;run<nbins;run++) {
      if(candidates[run]->ene > candidates[cur]->ene)
        CAND_SWAP(candidates[run], candidates[cur]);
    }
  }
}


void aubio_pitchmcomb_sort_cand_freq(aubio_spectralcandidate_t ** candidates, uint_t nbins) {
  uint_t cur = 0;
  uint_t run = 0;
  for (cur=0;cur<nbins;cur++) {
    run = cur + 1;
    for (run=cur;run<nbins;run++) {
      if(candidates[run]->ebin < candidates[cur]->ebin)
        CAND_SWAP(candidates[run], candidates[cur]);
    }
  }
}

aubio_pitchmcomb_t * new_aubio_pitchmcomb(uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate) {
  aubio_pitchmcomb_t * p = AUBIO_NEW(aubio_pitchmcomb_t);
  /* bug: should check if size / 8 > post+pre+1 */
  uint_t i;
  uint_t spec_size;
  p->spec_partition   = 4;
  p->ncand            = 5;
  p->npartials        = 5;
  p->cutoff           = 1.;
  p->threshold        = 0.01;
  p->win_post         = 8;
  p->win_pre          = 7;
  p->tau              = samplerate/bufsize;
  p->alpha            = 9.;
  p->goodcandidate    = 0;
  p->phasefreq        = bufsize/hopsize/TWO_PI;
  p->phasediff        = TWO_PI*hopsize/bufsize;
  spec_size = bufsize/p->spec_partition;
  //p->pickerfn = quadpick;
  //p->biquad = new_biquad(0.1600,0.3200,0.1600, -0.5949, 0.2348);
  /* allocate temp memory */
  p->newmag     = new_fvec(spec_size,channels);
  /* array for median */
  p->scratch    = new_fvec(spec_size,channels);
  /* array for phase */
  p->theta      = new_fvec(spec_size,channels);
  /* array for adaptative threshold */
  p->scratch2   = new_fvec(p->win_post+p->win_pre+1,channels);
  /* array of spectral peaks */
  p->peaks      = AUBIO_ARRAY(aubio_spectralpeak_t,spec_size);
  /* array of pointers to spectral candidates */
  p->candidates = AUBIO_ARRAY(aubio_spectralcandidate_t *,p->ncand);
  for (i=0;i<p->ncand;i++) {
    p->candidates[i] = AUBIO_NEW(aubio_spectralcandidate_t);
    p->candidates[i]->ecomb = AUBIO_ARRAY(smpl_t, spec_size);
  }
  return p;
}


void del_aubio_pitchmcomb (aubio_pitchmcomb_t *p) {
  uint_t i;
  del_fvec(p->newmag);
  del_fvec(p->scratch);
  del_fvec(p->scratch2);
  AUBIO_FREE(p->peaks);
  for (i=0;i<p->ncand;i++) {
    AUBIO_FREE(p->candidates[i]);
  }
  AUBIO_FREE(p->candidates);
  AUBIO_FREE(p);
}
