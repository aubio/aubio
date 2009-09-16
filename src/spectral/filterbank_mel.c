/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>
                      and Amaury Hazan <ahazan@iua.upf.edu>

  This file is part of Aubio.

  Aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "spectral/filterbank.h"
#include "mathutils.h"

void
aubio_filterbank_set_mel_coeffs (aubio_filterbank_t * fb, uint_t samplerate,
    smpl_t freq_min, smpl_t freq_max)
{

  fvec_t *filters = aubio_filterbank_get_coeffs (fb);
  uint_t n_filters = filters->channels, win_s = filters->length;

  //slaney params
  smpl_t lowestFrequency = 133.3333;
  smpl_t linearSpacing = 66.66666666;
  smpl_t logSpacing = 1.0711703;

  uint_t linearFilters = 13;
  uint_t logFilters = 27;
  uint_t allFilters = linearFilters + logFilters;

  //buffers for computing filter frequencies
  fvec_t *freqs = new_fvec (allFilters + 2, 1);

  fvec_t *lower_freqs = new_fvec (allFilters, 1);
  fvec_t *upper_freqs = new_fvec (allFilters, 1);
  fvec_t *center_freqs = new_fvec (allFilters, 1);

  fvec_t *triangle_heights = new_fvec (allFilters, 1);
  //lookup table of each bin frequency in hz
  fvec_t *fft_freqs = new_fvec (win_s, 1);

  uint_t filter_cnt, bin_cnt;

  //first step: filling all the linear filter frequencies
  for (filter_cnt = 0; filter_cnt < linearFilters; filter_cnt++) {
    freqs->data[0][filter_cnt] = lowestFrequency + filter_cnt * linearSpacing;
  }
  smpl_t lastlinearCF = freqs->data[0][filter_cnt - 1];

  //second step: filling all the log filter frequencies
  for (filter_cnt = 0; filter_cnt < logFilters + 2; filter_cnt++) {
    freqs->data[0][filter_cnt + linearFilters] =
        lastlinearCF * (pow (logSpacing, filter_cnt + 1));
  }

  //Option 1. copying interesting values to lower_freqs, center_freqs and upper freqs arrays
  //TODO: would be nicer to have a reference to freqs->data, anyway we do not care in this init step

  for (filter_cnt = 0; filter_cnt < allFilters; filter_cnt++) {
    lower_freqs->data[0][filter_cnt] = freqs->data[0][filter_cnt];
    center_freqs->data[0][filter_cnt] = freqs->data[0][filter_cnt + 1];
    upper_freqs->data[0][filter_cnt] = freqs->data[0][filter_cnt + 2];
  }

  //computing triangle heights so that each triangle has unit area
  for (filter_cnt = 0; filter_cnt < allFilters; filter_cnt++) {
    triangle_heights->data[0][filter_cnt] =
        2. / (upper_freqs->data[0][filter_cnt]
        - lower_freqs->data[0][filter_cnt]);
  }

  //AUBIO_DBG("filter tables frequencies\n");
  //for(filter_cnt=0; filter_cnt<allFilters; filter_cnt++)
  //  AUBIO_DBG("filter n. %d %f %f %f %f\n",
  //    filter_cnt, lower_freqs->data[0][filter_cnt], 
  //    center_freqs->data[0][filter_cnt], upper_freqs->data[0][filter_cnt], 
  //    triangle_heights->data[0][filter_cnt]);

  //filling the fft_freqs lookup table, which assigns the frequency in hz to each bin
  for (bin_cnt = 0; bin_cnt < win_s; bin_cnt++) {
    fft_freqs->data[0][bin_cnt] = aubio_bintofreq (bin_cnt, samplerate, win_s);
  }

  //building each filter table
  for (filter_cnt = 0; filter_cnt < allFilters; filter_cnt++) {

    //TODO:check special case : lower freq =0
    //calculating rise increment in mag/Hz
    smpl_t riseInc =
        triangle_heights->data[0][filter_cnt] /
        (center_freqs->data[0][filter_cnt] - lower_freqs->data[0][filter_cnt]);

    //zeroing begining of filter
    for (bin_cnt = 0; bin_cnt < win_s - 1; bin_cnt++) {
      filters->data[filter_cnt][bin_cnt] = 0.0;
      if (fft_freqs->data[0][bin_cnt] <= lower_freqs->data[0][filter_cnt] &&
          fft_freqs->data[0][bin_cnt + 1] > lower_freqs->data[0][filter_cnt]) {
        break;
      }
    }
    bin_cnt++;

    //positive slope
    for (; bin_cnt < win_s - 1; bin_cnt++) {
      filters->data[filter_cnt][bin_cnt] =
          (fft_freqs->data[0][bin_cnt] -
          lower_freqs->data[0][filter_cnt]) * riseInc;
      //if(fft_freqs->data[0][bin_cnt]<= center_freqs->data[0][filter_cnt] && fft_freqs->data[0][bin_cnt+1]> center_freqs->data[0][filter_cnt])
      if (fft_freqs->data[0][bin_cnt + 1] > center_freqs->data[0][filter_cnt])
        break;
    }
    //bin_cnt++;

    //negative slope
    for (; bin_cnt < win_s - 1; bin_cnt++) {

      //checking whether last value is less than 0...
      smpl_t val =
          triangle_heights->data[0][filter_cnt] - (fft_freqs->data[0][bin_cnt] -
          center_freqs->data[0][filter_cnt]) * riseInc;
      if (val >= 0)
        filters->data[filter_cnt][bin_cnt] = val;
      else
        filters->data[filter_cnt][bin_cnt] = 0.0;

      //if(fft_freqs->data[0][bin_cnt]<= upper_freqs->data[0][bin_cnt] && fft_freqs->data[0][bin_cnt+1]> upper_freqs->data[0][filter_cnt])
      //TODO: CHECK whether bugfix correct
      if (fft_freqs->data[0][bin_cnt + 1] > upper_freqs->data[0][filter_cnt])
        break;
    }
    //bin_cnt++;

    //zeroing tail
    for (; bin_cnt < win_s; bin_cnt++)
      filters->data[filter_cnt][bin_cnt] = 0.f;

  }

  /* destroy temporarly allocated vectors */
  del_fvec (freqs);
  del_fvec (lower_freqs);
  del_fvec (upper_freqs);
  del_fvec (center_freqs);

  del_fvec (triangle_heights);
  del_fvec (fft_freqs);

}
