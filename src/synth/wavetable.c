/*
  Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

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
#include "fmat.h"
#include "io/source.h"
#include "synth/wavetable.h"

#define WAVETABLE_LEN 4096

struct _aubio_wavetable_t {
  uint_t samplerate;
  uint_t blocksize;
  uint_t wavetable_length;
  fvec_t *wavetable;
  uint_t playing;
  smpl_t last_pos;

  smpl_t target_freq;
  smpl_t freq;
  smpl_t inc_freq;

  smpl_t target_amp;
  smpl_t amp;
  smpl_t inc_amp;
};

aubio_wavetable_t *new_aubio_wavetable(uint_t samplerate, uint_t blocksize)
{
  aubio_wavetable_t *s = AUBIO_NEW(aubio_wavetable_t);
  uint_t i = 0;
  s->samplerate = samplerate;
  s->blocksize = blocksize;
  s->wavetable_length = WAVETABLE_LEN;
  s->wavetable = new_fvec(s->wavetable_length + 3);
  for (i = 0; i < s->wavetable_length; i++) {
    s->wavetable->data[i] = SIN(TWO_PI * i / (smpl_t) s->wavetable_length );
  }
  s->wavetable->data[s->wavetable_length] = s->wavetable->data[0];
  s->wavetable->data[s->wavetable_length + 1] = s->wavetable->data[1];
  s->wavetable->data[s->wavetable_length + 2] = s->wavetable->data[2];
  s->playing = 0;
  s->last_pos = 0.;
  s->freq = 0.;
  s->target_freq = 0.;
  s->inc_freq = 0.;

  s->amp = 0.;
  s->target_amp = 0.;
  s->inc_amp = 0.;
  return s;
}

static smpl_t interp_2(fvec_t *input, smpl_t pos) {
  uint_t idx = (uint_t)FLOOR(pos);
  smpl_t frac = pos - (smpl_t)idx;
  smpl_t a = input->data[idx];
  smpl_t b = input->data[idx + 1];
  return a + frac * ( b - a );
}

void aubio_wavetable_do ( aubio_wavetable_t * s, fvec_t * input, fvec_t * output)
{
  uint_t i;
  if (s->playing) {
    smpl_t pos = s->last_pos;
    for (i = 0; i < output->length; i++) {
      if (s->freq != s->target_freq)
        s->freq += s->inc_freq;
      smpl_t inc = s->freq * (smpl_t)(s->wavetable_length) / (smpl_t) (s->samplerate);
      pos += inc;
      while (pos > s->wavetable_length) {
        pos -= s->wavetable_length;
      }
      if ( ABS(s->amp - s->target_amp) > ABS(s->inc_amp) )
        s->amp += s->inc_amp;
      else
        s->amp = s->target_amp;
      output->data[i] = s->amp * interp_2(s->wavetable, pos);
    }
    s->last_pos = pos;
  } else {
    fvec_set(output, 0.);
  }
  // add input to output if needed
  if (input && input != output) {
    for (i = 0; i < output->length; i++) {
      output->data[i] += input->data[i];
    }
  }
}

void aubio_wavetable_do_multi ( aubio_wavetable_t * s, fmat_t * input, fmat_t * output)
{
  uint_t i, j;
  if (s->playing) {
    smpl_t pos = s->last_pos;
    for (j = 0; j < output->length; j++) {
      if (s->freq != s->target_freq)
        s->freq += s->inc_freq;
      smpl_t inc = s->freq * (smpl_t)(s->wavetable_length) / (smpl_t) (s->samplerate);
      pos += inc;
      while (pos > s->wavetable_length) {
        pos -= s->wavetable_length;
      }
      for (i = 0; i < output->height; i++) {
        output->data[i][j] = interp_2(s->wavetable, pos);
      }
    }
    s->last_pos = pos;
  } else {
    for (j = 0; j < output->length; j++) {
      if (s->freq != s->target_freq)
        s->freq += s->inc_freq;
    }
    fmat_set(output, 0.);
  }
  // add output to input if needed
  if (input && input != output) {
    for (i = 0; i < output->height; i++) {
      for (j = 0; j < output->length; j++) {
        output->data[i][j] += input->data[i][j];
      }
    }
  }
}

uint_t aubio_wavetable_get_playing ( aubio_wavetable_t * s )
{
  return s->playing;
}

uint_t aubio_wavetable_set_playing ( aubio_wavetable_t * s, uint_t playing )
{
  s->playing = (playing == 1) ? 1 : 0;
  return 0;
}

uint_t aubio_wavetable_play ( aubio_wavetable_t * s )
{
  aubio_wavetable_set_amp (s, 0.7);
  return aubio_wavetable_set_playing (s, 1);
}

uint_t aubio_wavetable_stop ( aubio_wavetable_t * s )
{
  //aubio_wavetable_set_freq (s, 0.);
  aubio_wavetable_set_amp (s, 0.);
  //s->last_pos = 0;
  return aubio_wavetable_set_playing (s, 1);
}

uint_t aubio_wavetable_set_freq ( aubio_wavetable_t * s, smpl_t freq )
{
  if (freq >= 0 && freq < s->samplerate / 2.) {
    uint_t steps = 10;
    s->inc_freq = (freq - s->freq) / steps; 
    s->target_freq = freq;
    return 0;
  } else {
    return 1;
  }
}

smpl_t aubio_wavetable_get_freq ( aubio_wavetable_t * s) {
  return s->freq;
}

uint_t aubio_wavetable_set_amp ( aubio_wavetable_t * s, smpl_t amp )
{
  AUBIO_MSG("amp: %f, s->amp: %f, target_amp: %f, inc_amp: %f\n",
      amp, s->amp, s->target_amp, s->inc_amp);
  if (amp >= 0. && amp < 1.) {
    uint_t steps = 100;
    s->inc_amp = (amp - s->amp) / steps; 
    s->target_amp = amp;
    AUBIO_ERR("amp: %f, s->amp: %f, target_amp: %f, inc_amp: %f\n",
        amp, s->amp, s->target_amp, s->inc_amp);
    return 0;
  } else {
    return 1;
  }
}

smpl_t aubio_wavetable_get_amp ( aubio_wavetable_t * s) {
  return s->amp;
}

void del_aubio_wavetable( aubio_wavetable_t * s )
{
  del_fvec(s->wavetable);
  AUBIO_FREE(s);
}
