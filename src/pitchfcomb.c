/*
   Copyright (C) 2004, 2005  Mario Lang <mlang@delysid.org>

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
#include "fft.h"
#include "pitchfcomb.h"

#define MAX_PEAKS 8

typedef struct {
  smpl_t freq;
  smpl_t db;
} aubio_fpeak_t;

struct _aubio_pitchfcomb_t {
        uint_t fftSize;
        uint_t stepSize;
        uint_t rate;
	fvec_t * winput;
	fvec_t * win;
        cvec_t * fftOut;
        fvec_t * fftLastPhase;
	aubio_mfft_t * fft;
        //aubio_pvoc_t * pvoc;
};

aubio_pitchfcomb_t * new_aubio_pitchfcomb (uint_t bufsize, uint_t hopsize, uint_t samplerate)
{
  aubio_pitchfcomb_t * p = AUBIO_NEW(aubio_pitchfcomb_t);
  p->rate         = samplerate;
  p->fftSize      = bufsize;
  p->stepSize     = hopsize; 
  p->winput       = new_fvec(bufsize,1);
  p->fftOut       = new_cvec(bufsize,1);
  p->fftLastPhase = new_fvec(bufsize,1);
  p->fft = new_aubio_mfft(bufsize, 1);
  p->win = new_fvec(bufsize,1);
  aubio_window(p->win->data[0], bufsize, aubio_win_hanning);
  return p;
}

/* input must be stepsize long */
smpl_t aubio_pitchfcomb_detect (aubio_pitchfcomb_t * p, fvec_t * input)
{
  uint_t k, l, maxharm = 0;
  smpl_t freqPerBin = p->rate/(smpl_t)p->fftSize,
    phaseDifference = TWO_PI*(smpl_t)p->stepSize/(smpl_t)p->fftSize;
  aubio_fpeak_t peaks[MAX_PEAKS];

  for (k=0; k<MAX_PEAKS; k++) {
    peaks[k].db = -200.;
    peaks[k].freq = 0.;
  }

  for (k=0; k < input->length; k++){
	  p->winput->data[0][k] = p->win->data[0][k] * input->data[0][k];
  }
  aubio_mfft_do(p->fft,p->winput,p->fftOut);

  for (k=0; k<=p->fftSize/2; k++) {
    smpl_t
      magnitude = 20.*LOG10(2.*p->fftOut->norm[0][k]/(smpl_t)p->fftSize),
      phase     = p->fftOut->phas[0][k],
      tmp, freq;

    /* compute phase difference */
    tmp = phase - p->fftLastPhase->data[0][k];
    p->fftLastPhase->data[0][k] = phase;

    /* subtract expected phase difference */
    tmp -= (smpl_t)k*phaseDifference;

    /* map delta phase into +/- Pi interval */
    tmp = aubio_unwrap2pi(tmp);

    /* get deviation from bin frequency from the +/- Pi interval */
    tmp = p->fftSize/(smpl_t)p->stepSize*tmp/(TWO_PI);

    /* compute the k-th partials' true frequency */
    freq = (smpl_t)k*freqPerBin + tmp*freqPerBin;

    if (freq > 0.0 && magnitude > peaks[0].db) { // && magnitude < 0) {
      memmove(peaks+1, peaks, sizeof(aubio_fpeak_t)*(MAX_PEAKS-1));
      peaks[0].freq = freq;
      peaks[0].db = magnitude;
    }
  }
  
  k = 0;
  for (l=1; l<MAX_PEAKS && peaks[l].freq > 0.0; l++) {
    sint_t harmonic;
    for (harmonic=5; harmonic>1; harmonic--) {
      if (peaks[0].freq / peaks[l].freq < harmonic+.02 &&
  	peaks[0].freq / peaks[l].freq > harmonic-.02) {
        if (harmonic > (sint_t)maxharm &&
  	  peaks[0].db < peaks[l].db/2) {
          maxharm = harmonic;
  	  k = l;
        }
      }
    }
  }
  /* quick hack to clean output a bit */
  if (peaks[k].freq > 5000.) return 0.;
  return peaks[k].freq;
}

void del_aubio_pitchfcomb (aubio_pitchfcomb_t * p)
{
  del_cvec(p->fftOut);
  del_fvec(p->fftLastPhase);
  del_fvec(p->win);
  del_fvec(p->winput);
  del_aubio_mfft(p->fft);
  AUBIO_FREE(p);
}

