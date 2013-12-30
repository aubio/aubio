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

#define AUBIO_UNSTABLE 1 // for fvec_median
#include "utils.h"
#define PROG_HAS_PITCH 1
#define PROG_HAS_ONSET 1
#define PROG_HAS_JACK 1
// TODO add PROG_HAS_OUTPUT
#include "parse_args.h"

uint_t median = 6;

fvec_t *note_buffer;
fvec_t *note_buffer2;

smpl_t curnote = 0.;
smpl_t newnote = 0.;
uint_t isready = 0;

aubio_pitch_t *pitch;
aubio_onset_t *o;
fvec_t *onset;
fvec_t *pitch_obuf;

/** append new note candidate to the note_buffer and return filtered value. we
 * need to copy the input array as fvec_median destroy its input data.*/
void note_append (fvec_t * note_buffer, smpl_t curnote);
uint_t get_note (fvec_t * note_buffer, fvec_t * note_buffer2);

void process_block (fvec_t *ibuf, fvec_t *obuf)
{
  smpl_t new_pitch, curlevel;
  fvec_zeros(obuf);
  aubio_onset_do(o, ibuf, onset);

  aubio_pitch_do (pitch, ibuf, pitch_obuf);
  new_pitch = fvec_get_sample(pitch_obuf, 0);
  if(median){
    note_append(note_buffer, new_pitch);
  }

  /* curlevel is negatif or 1 if silence */
  curlevel = aubio_level_detection(ibuf, silence_threshold);
  if (fvec_get_sample(onset, 0)) {
    /* test for silence */
    if (curlevel == 1.) {
      if (median) isready = 0;
      /* send note off */
      send_noteon(curnote,0);
    } else {
      if (median) {
        isready = 1;
      } else {
        /* kill old note */
        send_noteon(curnote,0);
        /* get and send new one */
        send_noteon(new_pitch,127+(int)floor(curlevel));
        curnote = new_pitch;
      }
    }
  } else {
    if (median) {
      if (isready > 0)
        isready++;
      if (isready == median)
      {
        /* kill old note */
        send_noteon(curnote,0);
        newnote = get_note(note_buffer, note_buffer2);
        curnote = newnote;
        /* get and send new one */
        if (curnote>45){
          send_noteon(curnote,127+(int)floor(curlevel));
        }
      }
    } // if median
  }
}

void process_print (void)
{
  //if (verbose) outmsg("%f\n",pitch_obuf->data[0]);
}

void
note_append (fvec_t * note_buffer, smpl_t curnote)
{
  uint_t i = 0;
  for (i = 0; i < note_buffer->length - 1; i++) {
    note_buffer->data[i] = note_buffer->data[i + 1];
  }
  note_buffer->data[note_buffer->length - 1] = curnote;
  return;
}

uint_t
get_note (fvec_t * note_buffer, fvec_t * note_buffer2)
{
  uint_t i;
  for (i = 0; i < note_buffer->length; i++) {
    note_buffer2->data[i] = note_buffer->data[i];
  }
  return fvec_median (note_buffer2);
}

int main(int argc, char **argv) {
  examples_common_init(argc,argv);

  verbmsg ("using source: %s at %dHz\n", source_uri, samplerate);

  verbmsg ("onset method: %s, ", onset_method);
  verbmsg ("buffer_size: %d, ", buffer_size);
  verbmsg ("hop_size: %d, ", hop_size);
  verbmsg ("threshold: %f\n", onset_threshold);

  verbmsg ("pitch method: %s, ", pitch_method);
  verbmsg ("buffer_size: %d, ", buffer_size * 4);
  verbmsg ("hop_size: %d, ", hop_size);
  verbmsg ("tolerance: %f\n", pitch_tolerance);

  o = new_aubio_onset (onset_method, buffer_size, hop_size, samplerate);
  if (onset_threshold != 0.) aubio_onset_set_threshold (o, onset_threshold);
  onset = new_fvec (1);

  pitch = new_aubio_pitch (pitch_method, buffer_size * 4, hop_size, samplerate);
  if (pitch_tolerance != 0.) aubio_pitch_set_tolerance (pitch, pitch_tolerance);
  pitch_obuf = new_fvec (1);

  if (median) {
      note_buffer = new_fvec (median);
      note_buffer2 = new_fvec (median);
  }

  examples_common_process((aubio_process_func_t)process_block, process_print);

  // send a last note off
  send_noteon (curnote, 0);

  del_aubio_pitch (pitch);
  if (median) {
      del_fvec (note_buffer);
      del_fvec (note_buffer2);
  }
  del_fvec (pitch_obuf);

  examples_common_del();
  return 0;
}

