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

#include "config.h"
#include "aubio_priv.h"
#include "fvec.h"
#include "effects/pitchshift.h"

#ifdef HAVE_RUBBERBAND

#include "rubberband/rubberband-c.h"

// check rubberband is 1.8.1, warn if 1.3
#if !((RUBBERBAND_API_MAJOR_VERSION >= 2) && \
    (RUBBERBAND_API_MINOR_VERSION >= 5))
#warning RubberBandOptionDetectorSoft not available, \
 please upgrade rubberband to version 1.8.1 or higher
#define RubberBandOptionDetectorSoft 0x00000000
#endif

/** generic pitch shifting structure */
struct _aubio_pitchshift_t
{
  uint_t samplerate;              /**< samplerate */
  uint_t hopsize;                 /**< hop size */
  smpl_t timeratio;               /**< time ratio */
  smpl_t pitchscale;              /**< pitch scale */

  RubberBandState rb;
  RubberBandOptions rboptions;
};

aubio_pitchshift_t *
new_aubio_pitchshift (const char_t * mode,
    smpl_t pitchscale, uint_t hopsize, uint_t samplerate)
{
  aubio_pitchshift_t *p = AUBIO_NEW (aubio_pitchshift_t);
  p->samplerate = samplerate;
  p->hopsize = hopsize;
  p->timeratio = 1.;
  p->pitchscale = pitchscale;

  p->rboptions = RubberBandOptionProcessRealTime;

  if ( strcmp(mode,"crispness:0") == 0 ) {
    p->rboptions |= RubberBandOptionTransientsSmooth;
    p->rboptions |= RubberBandOptionWindowLong;
    p->rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:1") == 0 ) {
    p->rboptions |= RubberBandOptionDetectorSoft;
    p->rboptions |= RubberBandOptionTransientsSmooth;
    p->rboptions |= RubberBandOptionWindowLong;
    p->rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:2") == 0 ) {
    p->rboptions |= RubberBandOptionTransientsSmooth;
    p->rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:3") == 0 ) {
    p->rboptions |= RubberBandOptionTransientsSmooth;
  } else if ( strcmp(mode, "crispness:4") == 0 ) {
    // same as "default"
  } else if ( strcmp(mode, "crispness:5") == 0 ) {
    p->rboptions |= RubberBandOptionTransientsCrisp;
  } else if ( strcmp(mode, "crispness:6") == 0 ) {
    p->rboptions |= RubberBandOptionTransientsCrisp;
    p->rboptions |= RubberBandOptionWindowShort;
    p->rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "default") == 0 ) {
    // nothing to do
  } else {
    AUBIO_ERR("pitchshift: unknown pitch shifting method %s\n", mode);
    goto beach;
  }
  //AUBIO_MSG("pitchshift: using pitch shifting method %s\n", mode);

  //p->rboptions |= RubberBandOptionTransientsCrisp;
  //p->rboptions |= RubberBandOptionWindowStandard;
  //p->rboptions |= RubberBandOptionSmoothingOff;
  //p->rboptions |= RubberBandOptionFormantShifted;
  //p->rboptions |= RubberBandOptionPitchHighConsistency;
  p->rb = rubberband_new(samplerate, 1, p->rboptions, p->timeratio, p->pitchscale);
  rubberband_set_max_process_size(p->rb, p->hopsize);
  //rubberband_set_debug_level(p->rb, 10);

#if 1
  // warm up rubber band
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
  int available = rubberband_available(p->rb);
  fvec_t *zeros = new_fvec(p->hopsize);
  while (available <= latency) {
    rubberband_process(p->rb, (const float* const*)&(zeros->data), p->hopsize, 0);
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
  if (pitchscale >= 0.0625  && pitchscale <= 4.) {
    p->pitchscale = pitchscale;
    rubberband_set_pitch_scale(p->rb, p->pitchscale);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("pitchshift: could not set pitchscale to %.2f\n", pitchscale);
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
    AUBIO_ERR("pitchshift: could not set transpose to %.2f\n", transpose);
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
  int output = 0;
  rubberband_process(p->rb, (const float* const*)&(in->data), p->hopsize, output);
  if (rubberband_available(p->rb) >= (int)p->hopsize) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), p->hopsize);
  } else {
    AUBIO_WRN("pitchshift: catching up with zeros, only %d available, needed: %d, "
        "current pitchscale: %f\n",
        rubberband_available(p->rb), p->hopsize, p->pitchscale);
    fvec_zeros(out);
  }
}

#else

// TODO fallback pitch shifting implementation

struct _aubio_pitchshift_t
{
  void *dummy;
};

void aubio_pitchshift_do (aubio_pitchshift_t * o UNUSED, const fvec_t * in UNUSED,
    fvec_t * out UNUSED) {
}

void del_aubio_pitchshift (aubio_pitchshift_t * o UNUSED) {
}

aubio_pitchshift_t *new_aubio_pitchshift (const char_t * method UNUSED,
    smpl_t pitchscale UNUSED, uint_t hop_size UNUSED, uint_t samplerate UNUSED)
{
  AUBIO_ERR ("aubio was not compiled with rubberband\n");
  return NULL;
}

uint_t aubio_pitchshift_set_pitchscale (aubio_pitchshift_t * o UNUSED, smpl_t pitchscale UNUSED)
{
  return AUBIO_FAIL;
}

smpl_t aubio_pitchshift_get_pitchscale (aubio_pitchshift_t * o UNUSED)
{
  return 1.;
}

uint_t aubio_pitchshift_set_transpose (aubio_pitchshift_t * o UNUSED, smpl_t transpose UNUSED) {
  return AUBIO_FAIL;
}

smpl_t aubio_pitchshift_get_transpose (aubio_pitchshift_t * o UNUSED) {
  return 0.;
}

uint_t aubio_pitchshift_get_latency (aubio_pitchshift_t * o UNUSED) {
  return 0.;
}

// end of dummy implementation

#endif /* HAVE_RUBBERBAND */
