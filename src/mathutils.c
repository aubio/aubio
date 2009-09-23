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

/* see in mathutils.h for doc */

#include "aubio_priv.h"
#include "fvec.h"
#include "mathutils.h"
#include "config.h"

void aubio_window(smpl_t *w, uint_t size, aubio_window_type wintype) {
  uint_t i;
  switch(wintype) {
    case aubio_win_rectangle:
      for (i=0;i<size;i++)
        w[i] = 0.5;
      break;
    case aubio_win_hamming:
      for (i=0;i<size;i++)
        w[i] = 0.54 - 0.46 * COS(TWO_PI * i / (size));
      break;
    case aubio_win_hanning:
      for (i=0;i<size;i++)
        w[i] = 0.5 - (0.5 * COS(TWO_PI * i / (size)));
      break;
    case aubio_win_hanningz:
      for (i=0;i<size;i++)
        w[i] = 0.5 * (1.0 - COS(TWO_PI * i / (size)));
      break;
    case aubio_win_blackman:
      for (i=0;i<size;i++)
        w[i] = 0.42
          - 0.50 * COS(    TWO_PI*i/(size-1.0))
          + 0.08 * COS(2.0*TWO_PI*i/(size-1.0));
      break;
    case aubio_win_blackman_harris:
      for (i=0;i<size;i++)
        w[i] = 0.35875
          - 0.48829 * COS(    TWO_PI*i/(size-1.0))
          + 0.14128 * COS(2.0*TWO_PI*i/(size-1.0))
          - 0.01168 * COS(3.0*TWO_PI*i/(size-1.0));
      break;
    case aubio_win_gaussian:
      for (i=0;i<size;i++)
        w[i] = EXP(- 1.0 / SQR(size) * SQR(2.0*i-size));
      break;
    case aubio_win_welch:
      for (i=0;i<size;i++)
        w[i] = 1.0 - SQR((2*i-size)/(size+1.0));
      break;
    case aubio_win_parzen:
      for (i=0;i<size;i++)
        w[i] = 1.0 - ABS((2*i-size)/(size+1.0));
      break;
    default:
      break;
  }
}

smpl_t aubio_unwrap2pi(smpl_t phase) {
  /* mod(phase+pi,-2pi)+pi */
  return phase + TWO_PI * (1. + FLOOR(-(phase+PI)/TWO_PI));
}

smpl_t vec_mean(fvec_t *s) {
  uint_t i,j;
  smpl_t tmp = 0.0f;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++)
      tmp += s->data[i][j];
  return tmp/(smpl_t)(s->length);
}

smpl_t vec_sum(fvec_t *s) {
  uint_t i,j;
  smpl_t tmp = 0.0f;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++)
      tmp += s->data[i][j];
  return tmp;
}

smpl_t vec_max(fvec_t *s) {
  uint_t i,j;
  smpl_t tmp = 0.0f;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++)
      tmp = (tmp > s->data[i][j])? tmp : s->data[i][j];
  return tmp;
}

smpl_t vec_min(fvec_t *s) {
  uint_t i,j;
  smpl_t tmp = s->data[0][0];
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++)
      tmp = (tmp < s->data[i][j])? tmp : s->data[i][j]  ;
  return tmp;
}

uint_t vec_min_elem(fvec_t *s) {
  uint_t i,j=0, pos=0.;
  smpl_t tmp = s->data[0][0];
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++) {
      pos = (tmp < s->data[i][j])? pos : j;
      tmp = (tmp < s->data[i][j])? tmp : s->data[i][j]  ;
    }
  return pos;
}

uint_t vec_max_elem(fvec_t *s) {
  uint_t i,j=0, pos=0.;
  smpl_t tmp = 0.0f;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length; j++) {
      pos = (tmp > s->data[i][j])? pos : j;
      tmp = (tmp > s->data[i][j])? tmp : s->data[i][j]  ;
    }
  return pos;
}

void vec_shift(fvec_t *s) {
  uint_t i,j;
  //smpl_t tmp = 0.0f;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->length / 2 ; j++) {
      //tmp = s->data[i][j];
      //s->data[i][j] = s->data[i][j+s->length/2];
      //s->data[i][j+s->length/2] = tmp;
      ELEM_SWAP(s->data[i][j],s->data[i][j+s->length/2]);
    }
}

smpl_t vec_local_energy(fvec_t * f) {
  smpl_t locE = 0.;
  uint_t i,j;
  for (i=0;i<f->channels;i++)
    for (j=0;j<f->length;j++)
      locE+=SQR(f->data[i][j]);
  return locE;
}

smpl_t vec_local_hfc(fvec_t * f) {
  smpl_t locE = 0.;
  uint_t i,j;
  for (i=0;i<f->channels;i++)
    for (j=0;j<f->length;j++)
      locE+=(i+1)*f->data[i][j];
  return locE;
}

smpl_t vec_alpha_norm(fvec_t * DF, smpl_t alpha) {
  smpl_t tmp = 0.;
  uint_t i,j;
  for (i=0;i<DF->channels;i++)
    for (j=0;j<DF->length;j++)
      tmp += POW(ABS(DF->data[i][j]),alpha);
  return POW(tmp/DF->length,1./alpha);
}

void vec_dc_removal(fvec_t * mag) {
    smpl_t mini = 0.;
    uint_t length = mag->length, i=0, j;
    mini = vec_min(mag);
    for (j=0;j<length;j++) {
      mag->data[i][j] -= mini;
    }
}

void vec_alpha_normalise(fvec_t * mag, uint_t alpha) {
  smpl_t alphan = 1.;
  uint_t length = mag->length, i=0, j;
  alphan = vec_alpha_norm(mag,alpha);
  for (j=0;j<length;j++){
    mag->data[i][j] /= alphan;
  }
}

void vec_add(fvec_t * mag, smpl_t threshold) {
  uint_t length = mag->length, i=0, j;
  for (j=0;j<length;j++) {
    mag->data[i][j] += threshold;
  }
}

void vec_adapt_thres(fvec_t * vec, fvec_t * tmp,
    uint_t post, uint_t pre) {
  uint_t length = vec->length, i=0, j;
  for (j=0;j<length;j++) {
    vec->data[i][j] -= vec_moving_thres(vec, tmp, post, pre, j);
  }
}

smpl_t vec_moving_thres(fvec_t * vec, fvec_t * tmpvec,
    uint_t post, uint_t pre, uint_t pos) {
  smpl_t * medar = (smpl_t *)tmpvec->data[0];
  uint_t k;
  uint_t win_length =  post+pre+1;
  uint_t length =  vec->length;
  /* post part of the buffer does not exist */
  if (pos<post+1) {
    for (k=0;k<post+1-pos;k++)
      medar[k] = 0.; /* 0-padding at the beginning */
    for (k=post+1-pos;k<win_length;k++)
      medar[k] = vec->data[0][k+pos-post];
  /* the buffer is fully defined */
  } else if (pos+pre<length) {
    for (k=0;k<win_length;k++)
      medar[k] = vec->data[0][k+pos-post];
  /* pre part of the buffer does not exist */
  } else {
    for (k=0;k<length-pos+post;k++)
      medar[k] = vec->data[0][k+pos-post];
    for (k=length-pos+post;k<win_length;k++)
      medar[k] = 0.; /* 0-padding at the end */
  }
  return vec_median(tmpvec);
}

smpl_t vec_median(fvec_t * input) {
  uint_t n = input->length;
  smpl_t * arr = (smpl_t *) input->data[0];
  uint_t low, high ;
  uint_t median;
  uint_t middle, ll, hh;

  low = 0 ; high = n-1 ; median = (low + high) / 2;
  for (;;) {
    if (high <= low) /* One element only */
      return arr[median] ;

    if (high == low + 1) {  /* Two elements only */
      if (arr[low] > arr[high])
        ELEM_SWAP(arr[low], arr[high]) ;
      return arr[median] ;
    }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]);
    if (arr[low]    > arr[high])    ELEM_SWAP(arr[low],    arr[high]);
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
      do ll++; while (arr[low] > arr[ll]) ;
      do hh--; while (arr[hh]  > arr[low]) ;

      if (hh < ll)
        break;

      ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}

smpl_t vec_quadint(fvec_t * x,uint_t pos, uint_t span) {
  smpl_t s0, s1, s2;
  uint_t x0 = (pos < span) ? pos : pos - span;
  uint_t x2 = (pos + span < x->length) ? pos + span : pos;
  if (x0 == pos) return (x->data[0][pos] <= x->data[0][x2]) ? pos : x2;
  if (x2 == pos) return (x->data[0][pos] <= x->data[0][x0]) ? pos : x0;
  s0 = x->data[0][x0];
  s1 = x->data[0][pos]     ;
  s2 = x->data[0][x2];
  return pos + 0.5 * (s2 - s0 ) / (s2 - 2.* s1 + s0);
}

smpl_t aubio_quadfrac(smpl_t s0, smpl_t s1, smpl_t s2, smpl_t pf) {
  smpl_t tmp = s0 + (pf/2.) * (pf * ( s0 - 2.*s1 + s2 ) - 3.*s0 + 4.*s1 - s2);
  return tmp;
}

uint_t vec_peakpick(fvec_t * onset, uint_t pos) {
  uint_t i=0, tmp=0;
  /*for (i=0;i<onset->channels;i++)*/
  tmp = (onset->data[i][pos] > onset->data[i][pos-1]
      &&  onset->data[i][pos] > onset->data[i][pos+1]
      &&  onset->data[i][pos] > 0.);
  return tmp;
}

smpl_t aubio_freqtomidi(smpl_t freq) {
  /* log(freq/A-2)/log(2) */
  smpl_t midi = freq/6.875;
  midi = LOG(midi)/0.69314718055995;
  midi *= 12;
  midi -= 3;
  return midi;
}

smpl_t aubio_miditofreq(smpl_t midi) {
  smpl_t freq = (midi+3.)/12.;
  freq = EXP(freq*0.69314718055995);
  freq *= 6.875;
  return freq;
}

smpl_t aubio_bintofreq(smpl_t bin, smpl_t samplerate, smpl_t fftsize) {
  smpl_t freq = samplerate/fftsize;
  return freq*bin;
}

smpl_t aubio_bintomidi(smpl_t bin, smpl_t samplerate, smpl_t fftsize) {
  smpl_t midi = aubio_bintofreq(bin,samplerate,fftsize);
  return aubio_freqtomidi(midi);
}

smpl_t aubio_freqtobin(smpl_t freq, smpl_t samplerate, smpl_t fftsize) {
  smpl_t bin = fftsize/samplerate;
  return freq*bin;
}

smpl_t aubio_miditobin(smpl_t midi, smpl_t samplerate, smpl_t fftsize) {
  smpl_t freq = aubio_miditofreq(midi);
  return aubio_freqtobin(freq,samplerate,fftsize);
}

/** returns 1 if wassilence is 0 and RMS(ibuf)<threshold
 * \bug mono
 */
uint_t aubio_silence_detection(fvec_t * ibuf, smpl_t threshold) {
  smpl_t loudness = 0;
  uint_t i=0,j;
  for (j=0;j<ibuf->length;j++) {
    loudness += SQR(ibuf->data[i][j]);
  }
  loudness = SQRT(loudness);
  loudness /= (smpl_t)ibuf->length;
  loudness = LIN2DB(loudness);

  return (loudness < threshold);
}

/** returns level log(RMS(ibuf)) if < threshold, 1 otherwise
 * \bug mono
 */
smpl_t aubio_level_detection(fvec_t * ibuf, smpl_t threshold) {
  smpl_t loudness = 0;
  uint_t i=0,j;
  for (j=0;j<ibuf->length;j++) {
    loudness += SQR(ibuf->data[i][j]);
  }
  loudness = SQRT(loudness);
  loudness /= (smpl_t)ibuf->length;
  loudness = LIN2DB(loudness);

  if (loudness < threshold)
    return 1.;
  else
    return loudness;
}

smpl_t aubio_zero_crossing_rate(fvec_t * input) {
  uint_t i=0,j;
  uint_t zcr = 0;
  for ( j = 1; j < input->length; j++ ) {
    // previous was strictly negative
    if( input->data[i][j-1] < 0. ) {
      // current is positive or null
      if ( input->data[i][j] >= 0. ) {
        zcr += 1;
      }
    // previous was positive or null
    } else {
      // current is strictly negative
      if ( input->data[i][j] < 0. ) {
        zcr += 1;
      }
    }
  }
  return zcr/(smpl_t)input->length;
}

void aubio_autocorr(fvec_t * input, fvec_t * output) {
  uint_t i = 0, j = 0, length = input->length;
  smpl_t * data = input->data[0];
  smpl_t * acf = output->data[0];
  smpl_t tmp =0.;
  for(i=0;i<length;i++){
    for(j=i;j<length;j++){
      tmp += data[j-i]*data[j];
    }
    acf[i] = tmp /(smpl_t)(length-i);
    tmp = 0.0;
  }
}

void aubio_cleanup(void) {
#if HAVE_FFTW3
  fftw_cleanup();
#else
#if HAVE_FFTW3F
  fftwf_cleanup();
#endif
#endif
}
