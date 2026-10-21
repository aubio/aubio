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
#include "types.h"
#include "fvec.h"
#include "lvec.h"
#include "temporal/filter.h"
#include "temporal/a_weighting.h"

/* The analog weighting curves of IEC 61672 (formerly IEC 651), with the
   rounded pole frequencies and 1 kHz gains of the long-standing adsgn and
   cdsgn designs, mapped to the samplerate with the bilinear transform. At
   8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000 and
   192000 Hz this gives the coefficients aubio used to ship as tables; any
   other samplerate works too. */
static const lsmp_t aubio_weighting_pi = 3.14159265358979323846;
static const lsmp_t aubio_weighting_f1 = 20.598997;
static const lsmp_t aubio_weighting_f2 = 107.65265;
static const lsmp_t aubio_weighting_f3 = 737.86223;
static const lsmp_t aubio_weighting_f4 = 12194.217;

uint_t
aubio_filter_set_a_weighting (aubio_filter_t * f, uint_t samplerate)
{
  uint_t order;

  if ((sint_t)samplerate <= 0) {
    AUBIO_ERROR("aubio_filter: failed setting A-weighting with samplerate %d\n", samplerate);
    return AUBIO_FAIL;
  }
  if (f == NULL) {
    AUBIO_ERROR("aubio_filter: failed setting A-weighting with filter NULL\n");
    return AUBIO_FAIL;
  }

  order = aubio_filter_get_order (f);
  if (order != 7) {
    AUBIO_ERROR ("aubio_filter: order of A-weighting filter must be 7, not %d\n", order);
    return AUBIO_FAIL;
  }

  aubio_filter_set_samplerate (f, samplerate);
  {
    lsmp_t w1 = 2. * aubio_weighting_pi * aubio_weighting_f1;
    lsmp_t w2 = 2. * aubio_weighting_pi * aubio_weighting_f2;
    lsmp_t w3 = 2. * aubio_weighting_pi * aubio_weighting_f3;
    lsmp_t w4 = 2. * aubio_weighting_pi * aubio_weighting_f4;
    /* H(s) = k s^4 / ((s + w1)^2 (s + w2) (s + w3) (s + w4)^2), coefficients
       by ascending power of s; k gives 1.9997 dB at 1 kHz */
    lsmp_t num[7] = { 0., 0., 0., 0., 0., 0., 0. };
    lsmp_t den[7] = { 0., 0., 0., 0., 0., 0., 0. };
    lsmp_t p14[5] = { 0., 0., 0., 0., 0. };
    lsmp_t q1[3] = { w1 * w1, 2. * w1, 1. };
    lsmp_t q4[3] = { w4 * w4, 2. * w4, 1. };
    lsmp_t p23[3] = { w2 * w3, w2 + w3, 1. };
    uint_t i, j;
    num[4] = w4 * w4 * pow(10., 1.9997 / 20.);
    for (i = 0; i < 3; i++)
      for (j = 0; j < 3; j++) p14[i + j] += q1[i] * q4[j];
    for (i = 0; i < 5; i++)
      for (j = 0; j < 3; j++) den[i + j] += p14[i] * p23[j];
    return aubio_filter_set_analog (f, num, den, 6);
  }
}

aubio_filter_t *
new_aubio_filter_a_weighting (uint_t samplerate)
{
  aubio_filter_t *f = new_aubio_filter (7);
  if (aubio_filter_set_a_weighting(f,samplerate) != AUBIO_OK) {
    del_aubio_filter(f);
    return NULL;
  }
  return f;
}
