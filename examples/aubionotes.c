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

#define AUBIO_UNSTABLE 1 // for fvec_median

#include "utils.h"

/* pitch objects */
smpl_t pitch = 0.;

uint_t median = 6;
smpl_t curlevel = 0.;

aubio_pitch_t *pitchdet;

fvec_t *note_buffer = NULL;
fvec_t *note_buffer2 = NULL;

smpl_t curnote = 0.;
smpl_t newnote = 0.;
uint_t isready = 0;
unsigned int pos = 0; /*frames%dspblocksize*/

aubio_pitch_t *pitchdet;
aubio_onset_t *o;
fvec_t *onset;
fvec_t *pitch_obuf;

/** append new note candidate to the note_buffer and return filtered value. we
 * need to copy the input array as fvec_median destroy its input data.*/
void note_append (fvec_t * note_buffer, smpl_t curnote);
uint_t get_note (fvec_t * note_buffer, fvec_t * note_buffer2);

static int aubio_process(smpl_t **input, smpl_t **output, int nframes) {
  unsigned int j;       /*frames*/
  for (j=0;j<(unsigned)nframes;j++) {
    if(usejack) {
      /* write input to datanew */
      fvec_write_sample(ibuf, input[0][j], pos);
      /* put synthnew in output */
      output[0][j] = fvec_read_sample(obuf, pos);
    }
    /*time for fft*/
    if (pos == overlap_size-1) {         
      /* block loop */
      aubio_onset_do(o, ibuf, onset);
      
      aubio_pitch_do (pitchdet, ibuf, pitch_obuf);
      pitch = fvec_read_sample(pitch_obuf, 0);
      if(median){
              note_append(note_buffer, pitch);
      }

      /* curlevel is negatif or 1 if silence */
      curlevel = aubio_level_detection(ibuf, silence);
      if (fvec_read_sample(onset, 0)) {
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
                              send_noteon(pitch,127+(int)floor(curlevel));
                              curnote = pitch;
                      }

                      for (pos = 0; pos < overlap_size; pos++){
                              obuf->data[pos] = woodblock->data[pos];
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
        for (pos = 0; pos < overlap_size; pos++)
          obuf->data[pos] = 0.;
      }
      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

static void process_print (void) {
      if (verbose) outmsg("%f\n",pitch);
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

  o = new_aubio_onset (onset_mode, buffer_size, overlap_size, samplerate);
  if (threshold != 0.) aubio_onset_set_threshold (o, threshold);
  onset = new_fvec (1);

  pitchdet = new_aubio_pitch (pitch_mode, buffer_size * 4,
          overlap_size, samplerate);
  aubio_pitch_set_tolerance (pitchdet, 0.7);
  pitch_obuf = new_fvec (1);
  if (median) {
      note_buffer = new_fvec (median);
      note_buffer2 = new_fvec (median);
  }

  examples_common_process(aubio_process, process_print);

  send_noteon (curnote, 0);
  del_aubio_pitch (pitchdet);
  if (median) {
      del_fvec (note_buffer);
      del_fvec (note_buffer2);
  }
  del_fvec (pitch_obuf);

  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  return 0;
}

