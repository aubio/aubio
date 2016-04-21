/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>
                      and Amaury Hazan <ahazan@iua.upf.edu>

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
#include "fmat.h"
#include "fvec.h"
#include "cvec.h"
#include "spectral/filterbank.h"
#include "spectral/filterbank_mel.h"
#include "mathutils.h"

uint_t
aubio_filterbank_set_triangle_bands (aubio_filterbank_t * fb,
    const fvec_t * freqs, smpl_t samplerate)
{

  fmat_t *filters = aubio_filterbank_get_coeffs (fb);
  uint_t n_filters = filters->height, win_s = filters->length;
  fvec_t *lower_freqs, *upper_freqs, *center_freqs;
  fvec_t *triangle_heights, *fft_freqs;

  uint_t fn;                    /* filter counter */
  uint_t bin;                   /* bin counter */

  smpl_t riseInc, downInc;

  /* freqs define the bands of triangular overlapping windows.
     throw a warning if filterbank object fb is too short. */
  if (freqs->length - 2 > n_filters) {
    AUBIO_WRN ("not enough filters, %d allocated but %d requested\n",
        n_filters, freqs->length - 2);
  }

  if (freqs->length - 2 < n_filters) {
    AUBIO_WRN ("too many filters, %d allocated but %d requested\n",
        n_filters, freqs->length - 2);
  }

  if (freqs->data[freqs->length - 1] > samplerate / 2) {
    AUBIO_WRN ("Nyquist frequency is %fHz, but highest frequency band ends at \
%fHz\n", samplerate / 2, freqs->data[freqs->length - 1]);
  }

  /* convenience reference to lower/center/upper frequency for each triangle */
  lower_freqs = new_fvec (n_filters);
  upper_freqs = new_fvec (n_filters);
  center_freqs = new_fvec (n_filters);

  /* height of each triangle */
  triangle_heights = new_fvec (n_filters);

  /* lookup table of each bin frequency in hz */
  fft_freqs = new_fvec (win_s);

  /* fill up the lower/center/upper */
  for (fn = 0; fn < n_filters; fn++) {
    lower_freqs->data[fn] = freqs->data[fn];
    center_freqs->data[fn] = freqs->data[fn + 1];
    upper_freqs->data[fn] = freqs->data[fn + 2];
  }

  /* compute triangle heights so that each triangle has unit area */
  for (fn = 0; fn < n_filters; fn++) {
    triangle_heights->data[fn] =
        2. / (upper_freqs->data[fn] - lower_freqs->data[fn]);
  }

  /* fill fft_freqs lookup table, which assigns the frequency in hz to each bin */
  for (bin = 0; bin < win_s; bin++) {
    fft_freqs->data[bin] =
        aubio_bintofreq (bin, samplerate, (win_s - 1) * 2);
  }

  /* zeroing of all filters */
  fmat_zeros (filters);

  if (fft_freqs->data[1] >= lower_freqs->data[0]) {
    /* - 1 to make sure we don't miss the smallest power of two */
    uint_t min_win_s =
        (uint_t) FLOOR (samplerate / lower_freqs->data[0]) - 1;
    AUBIO_WRN ("Lowest frequency bin (%.2fHz) is higher than lowest frequency \
band (%.2f-%.2fHz). Consider increasing the window size from %d to %d.\n",
        fft_freqs->data[1], lower_freqs->data[0],
        upper_freqs->data[0], (win_s - 1) * 2,
        aubio_next_power_of_two (min_win_s));
  }

  /* building each filter table */
  for (fn = 0; fn < n_filters; fn++) {

    /* skip first elements */
    for (bin = 0; bin < win_s - 1; bin++) {
      if (fft_freqs->data[bin] <= lower_freqs->data[fn] &&
          fft_freqs->data[bin + 1] > lower_freqs->data[fn]) {
        bin++;
        break;
      }
    }

    /* compute positive slope step size */
    riseInc =
        triangle_heights->data[fn] /
        (center_freqs->data[fn] - lower_freqs->data[fn]);

    /* compute coefficients in positive slope */
    for (; bin < win_s - 1; bin++) {
      filters->data[fn][bin] =
          (fft_freqs->data[bin] - lower_freqs->data[fn]) * riseInc;

      if (fft_freqs->data[bin + 1] >= center_freqs->data[fn]) {
        bin++;
        break;
      }
    }

    /* compute negative slope step size */
    downInc =
        triangle_heights->data[fn] /
        (upper_freqs->data[fn] - center_freqs->data[fn]);

    /* compute coefficents in negative slope */
    for (; bin < win_s - 1; bin++) {
      filters->data[fn][bin] +=
          (upper_freqs->data[fn] - fft_freqs->data[bin]) * downInc;

      if (filters->data[fn][bin] < 0.) {
        filters->data[fn][bin] = 0.;
      }

      if (fft_freqs->data[bin + 1] >= upper_freqs->data[fn])
        break;
    }
    /* nothing else to do */

  }

  /* destroy temporarly allocated vectors */
  del_fvec (lower_freqs);
  del_fvec (upper_freqs);
  del_fvec (center_freqs);

  del_fvec (triangle_heights);
  del_fvec (fft_freqs);

  return 0;
}

uint_t
aubio_filterbank_set_mel_coeffs_slaney (aubio_filterbank_t * fb,
    smpl_t samplerate)
{
  uint_t retval;

  /* Malcolm Slaney parameters */
  smpl_t lowestFrequency = 133.3333;
  smpl_t linearSpacing = 66.66666666;
  smpl_t logSpacing = 1.0711703;

  uint_t linearFilters = 13;
  uint_t logFilters = 27;
  uint_t n_filters = linearFilters + logFilters;

  uint_t fn;                    /* filter counter */

  smpl_t lastlinearCF;

  /* buffers to compute filter frequencies */
  fvec_t *freqs = new_fvec (n_filters + 2);

  /* first step: fill all the linear filter frequencies */
  for (fn = 0; fn < linearFilters; fn++) {
    freqs->data[fn] = lowestFrequency + fn * linearSpacing;
  }
  lastlinearCF = freqs->data[fn - 1];

  /* second step: fill all the log filter frequencies */
  for (fn = 0; fn < logFilters + 2; fn++) {
    freqs->data[fn + linearFilters] =
        lastlinearCF * (POW (logSpacing, fn + 1));
  }

  /* now compute the actual coefficients */
  retval = aubio_filterbank_set_triangle_bands (fb, freqs, samplerate);

  /* destroy vector used to store frequency limits */
  del_fvec (freqs);

  return retval;
}
