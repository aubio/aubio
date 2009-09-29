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


/* Requires lsmp_t to be long or double. float will NOT give reliable 
 * results */

#include "aubio_priv.h"
#include "fvec.h"
#include "lvec.h"
#include "mathutils.h"
#include "temporal/filter.h"

struct _aubio_filter_t
{
  uint_t order;
  uint_t samplerate;
  lvec_t *a;
  lvec_t *b;
  lvec_t *y;
  lvec_t *x;
};

void
aubio_filter_do_outplace (aubio_filter_t * f, fvec_t * in, fvec_t * out)
{
  fvec_copy (in, out);
  aubio_filter_do (f, out);
}

void
aubio_filter_do (aubio_filter_t * f, fvec_t * in)
{
  uint_t i, j, l, order = f->order;
  lsmp_t *x;
  lsmp_t *y;
  lsmp_t *a = f->a->data[0];
  lsmp_t *b = f->b->data[0];

  for (i = 0; i < in->channels; i++) {
    x = f->x->data[i];
    y = f->y->data[i];
    for (j = 0; j < in->length; j++) {
      /* new input */
      x[0] = KILL_DENORMAL (in->data[i][j]);
      y[0] = b[0] * x[0];
      for (l = 1; l < order; l++) {
        y[0] += b[l] * x[l];
        y[0] -= a[l] * y[l];
      }
      /* new output */
      in->data[i][j] = y[0];
      /* store for next sample */
      for (l = order - 1; l > 0; l--) {
        x[l] = x[l - 1];
        y[l] = y[l - 1];
      }
    }
    /* store for next run */
    f->x->data[i] = x;
    f->y->data[i] = y;
  }
}

/*  
 *
 * despite mirroring, end effects destroy both phse and amplitude. the longer
 * the buffer, the less affected they are.
 *
 * replacing with zeros clicks.
 *
 * seems broken for order > 4 (see biquad_do_filtfilt for audible one) 
 */
void aubio_filter_do_filtfilt(aubio_filter_t * f, fvec_t * in, fvec_t * tmp) {
  uint_t j,i=0;
  uint_t length = in->length;
  //uint_t order = f->order;
  //lsmp_t mir;
  /* mirroring */
  //mir = 2*in->data[i][0];
  //for (j=1;j<order;j++)
  //f->x[j] = 0.;//mir - in->data[i][order-j];
  /* apply filtering */
  aubio_filter_do(f,in);
  /* invert */
  for (j = 0; j < length; j++)
    tmp->data[i][length-j-1] = in->data[i][j];
  /* mirror inverted */
  //mir = 2*tmp->data[i][0];
  //for (j=1;j<order;j++)
  //f->x[j] = 0.;//mir - tmp->data[i][order-j];
  /* apply filtering on inverted */
  aubio_filter_do(f,tmp);
  /* invert back */
  for (j = 0; j < length; j++)
    in->data[i][j] = tmp->data[i][length-j-1];
}

lvec_t *
aubio_filter_get_feedback (aubio_filter_t * f)
{
  return f->a;
}

lvec_t *
aubio_filter_get_feedforward (aubio_filter_t * f)
{
  return f->b;
}

uint_t
aubio_filter_get_order (aubio_filter_t * f)
{
  return f->order;
}

uint_t
aubio_filter_get_samplerate (aubio_filter_t * f)
{
  return f->samplerate;
}

aubio_filter_t *
new_aubio_filter (uint_t samplerate, uint_t order, uint_t channels)
{
  aubio_filter_t *f = AUBIO_NEW (aubio_filter_t);
  f->x = new_lvec (order, channels);
  f->y = new_lvec (order, channels);
  f->a = new_lvec (order, 1);
  f->b = new_lvec (order, 1);
  f->samplerate = samplerate;
  f->order = order;
  /* set default to identity */
  f->a->data[0][1] = 1.;
  return f;
}

void
del_aubio_filter (aubio_filter_t * f)
{
  del_lvec (f->a);
  del_lvec (f->b);
  del_lvec (f->x);
  del_lvec (f->y);
  AUBIO_FREE (f);
  return;
}
