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
#include "fvec.h"
#include "cvec.h"
#include "spectral/filterbank.h"
#include "mathutils.h"

/** \brief A structure to store a set of n_filters filters of lenghts win_s */
struct _aubio_filterbank_t
{
  uint_t win_s;
  uint_t n_filters;
  fvec_t *filters;
};

aubio_filterbank_t *
new_aubio_filterbank (uint_t n_filters, uint_t win_s)
{
  /* allocate space for filterbank object */
  aubio_filterbank_t *fb = AUBIO_NEW (aubio_filterbank_t);
  fb->win_s = win_s;
  fb->n_filters = n_filters;

  /* allocate filter tables, an fvec of length win_s and of filter_cnt channel */
  fb->filters = new_fvec (win_s / 2 + 1, n_filters);

  return fb;
}

void
del_aubio_filterbank (aubio_filterbank_t * fb)
{
  del_fvec (fb->filters);
  AUBIO_FREE (fb);
}

void
aubio_filterbank_do (aubio_filterbank_t * f, cvec_t * in, fvec_t * out)
{
  uint_t i, j, fn;

  /* apply filter to all input channel, provided out has enough channels */
  uint_t max_channels = MIN (in->channels, out->channels);
  uint_t max_filters = MIN (f->n_filters, out->length);

  /* reset all values in output vector */
  fvec_zeros (out);

  /* apply filters on all channels */
  for (i = 0; i < max_channels; i++) {

    /* for each filter */
    for (fn = 0; fn < max_filters; fn++) {

      /* for each sample */
      for (j = 0; j < in->length; j++) {
        out->data[i][fn] += in->norm[i][j] * f->filters->data[fn][j];
      }
    }
  }

  return;
}

fvec_t *
aubio_filterbank_get_coeffs (aubio_filterbank_t * f)
{
  return f->filters;
}
