/*
  Copyright (C) 2014 Paul Brossier <piem@aubio.org>

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
#include "pitch/pitch.h"
#include "onset/onset.h"
#include "notes/notes.h"

struct _aubio_notes_t {

  uint_t onset_buf_size;
  uint_t pitch_buf_size;
  uint_t hop_size;

  uint_t samplerate;

  uint_t median;
  fvec_t *note_buffer;
  fvec_t *note_buffer2;

  aubio_pitch_t *pitch;
  aubio_onset_t *onset;
  fvec_t *onset_output;
  fvec_t *pitch_output;

  smpl_t curnote;
  smpl_t newnote;
};

aubio_notes_t * new_aubio_notes (char_t * notes_method,
    uint_t buf_size, uint_t hop_size, uint_t samplerate) {
  aubio_notes_t *o = AUBIO_NEW(aubio_notes_t);

  o->onset_buf_size = buf_size;
  o->pitch_buf_size = buf_size * 4;
  o->hop_size = hop_size;

  o->samplerate = samplerate;

  o->median = 9;

  if (strcmp(notes_method, "default") != 0) {
    AUBIO_ERR("unknown notes detection method %s, using default.\n",
       notes_method);
    goto fail;
  }
  o->note_buffer = new_fvec(o->median);
  o->note_buffer2 = new_fvec(o->median);

  o->curnote = -1.;
  o->newnote = -1.;

  return o;

fail:
  del_aubio_notes(o);
  return NULL;
}

void del_aubio_notes (aubio_notes_t *o) {
  if (o->note_buffer) del_fvec(o->note_buffer);
  if (o->note_buffer2) del_fvec(o->note_buffer2);
  AUBIO_FREE(o);
}
