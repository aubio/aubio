/*
  Copyright (C) 2016 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_RUBBERBAND

#include "fvec.h"
#include "effects/pitchshift.h"

#include <rubberband/rubberband-c.h>

/** generic pitch shifting structure */
struct _aubio_pitchshift_t
{
  uint_t samplerate;              /**< samplerate */
  uint_t hopsize;                 /**< hop size */
  smpl_t pitchscale;              /**< pitch scale */

  RubberBandState rb;
  RubberBandOptions rboptions;
};

extern RubberBandOptions aubio_get_rubberband_opts(const char_t *mode);

aubio_pitchshift_t *
new_aubio_pitchshift (const char_t * mode,
    smpl_t transpose, uint_t hopsize, uint_t samplerate)
{
  aubio_pitchshift_t *p = AUBIO_NEW (aubio_pitchshift_t);
  p->samplerate = samplerate;
  p->hopsize = hopsize;
  p->pitchscale = 1.;
  p->rb = NULL;
  if ((sint_t)hopsize <= 0) {
    AUBIO_ERR("pitchshift: hop_size should be >= 0, got %d\n", hopsize);
    goto beach;
  }
  if ((sint_t)samplerate <= 0) {
    AUBIO_ERR("pitchshift: samplerate should be >= 0, got %d\n", samplerate);
    goto beach;
  }

  p->rboptions = aubio_get_rubberband_opts(mode);
  if (p->rboptions < 0) {
    AUBIO_ERR("pitchshift: unknown pitch shifting method %s\n", mode);
    goto beach;
  }

  //AUBIO_MSG("pitchshift: using pitch shifting method %s\n", mode);

  p->rb = rubberband_new(samplerate, 1, p->rboptions, 1., p->pitchscale);
  rubberband_set_max_process_size(p->rb, p->hopsize);
  //rubberband_set_debug_level(p->rb, 10);

  if (aubio_pitchshift_set_transpose(p, transpose)) goto beach;

#if 1
  // warm up rubber band
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
  int available = rubberband_available(p->rb);
  fvec_t *zeros = new_fvec(p->hopsize);
  while (available <= (int)latency) {
    rubberband_process(p->rb,
        (const float* const*)&(zeros->data), p->hopsize, 0);
    available = rubberband_available(p->rb);
  }
  del_fvec(zeros);
#endif

  return p;

beach:
  del_aubio_pitchshift(p);
  return NULL;
}

void
del_aubio_pitchshift (aubio_pitchshift_t * p)
{
  if (p->rb) {
    rubberband_delete(p->rb);
  }
  AUBIO_FREE (p);
}

uint_t aubio_pitchshift_get_latency (aubio_pitchshift_t * p) {
  return rubberband_get_latency(p->rb);
}

uint_t
aubio_pitchshift_set_pitchscale (aubio_pitchshift_t * p, smpl_t pitchscale)
{
  if (pitchscale >= 0.25  && pitchscale <= 4.) {
    p->pitchscale = pitchscale;
    rubberband_set_pitch_scale(p->rb, p->pitchscale);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("pitchshift: could not set pitchscale to '%f',"
        " should be in the range [0.25, 4.].\n", pitchscale);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_pitchshift_get_pitchscale (aubio_pitchshift_t * p)
{
  return p->pitchscale;
}

uint_t
aubio_pitchshift_set_transpose(aubio_pitchshift_t * p, smpl_t transpose)
{
  if (transpose >= -24. && transpose <= 24.) {
    smpl_t pitchscale = POW(2., transpose / 12.);
    return aubio_pitchshift_set_pitchscale(p, pitchscale);
  } else {
    AUBIO_ERR("pitchshift: could not set transpose to '%f',"
        " should be in the range [-24; 24.].\n", transpose);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_pitchshift_get_transpose(aubio_pitchshift_t * p)
{
  return 12. * LOG(p->pitchscale) / LOG(2.0);
}

void
aubio_pitchshift_do (aubio_pitchshift_t * p, const fvec_t * in, fvec_t * out)
{
  // third parameter is always 0 since we are never expecting a final frame
  rubberband_process(p->rb, (const float* const*)&(in->data), p->hopsize, 0);
  if (rubberband_available(p->rb) >= (int)p->hopsize) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), p->hopsize);
  } else {
    AUBIO_WRN("pitchshift: catching up with zeros"
        ", only %d available, needed: %d, current pitchscale: %f\n",
        rubberband_available(p->rb), p->hopsize, p->pitchscale);
    fvec_zeros(out);
  }
}

#endif
