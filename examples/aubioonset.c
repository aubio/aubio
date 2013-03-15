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

#include "utils.h"

unsigned int pos = 0; /*frames%dspblocksize*/

aubio_onset_t *o;
fvec_t *onset;

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
      aubio_onset_do (o, ibuf, onset);
      if ( fvec_read_sample(onset, 0) ) {
        fvec_copy (woodblock, obuf);
      } else {
        fvec_zeros (obuf);
      }
      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

static void
process_print (void)
{
  /* output times in seconds, taking back some delay to ensure the label is
   * _before_ the actual onset */
  if (!verbose && usejack)
    return;
  smpl_t onset_found = fvec_read_sample (onset, 0);
  if (onset_found) {
    outmsg ("%f\n", aubio_onset_get_last_onset_s (o) );
  }
}

int main(int argc, char **argv) {
  frames_delay = 3;
  examples_common_init(argc,argv);

  o = new_aubio_onset (onset_mode, buffer_size, overlap_size, samplerate);
  if (threshold != 0.) aubio_onset_set_threshold (o, threshold);
  onset = new_fvec (1);

  examples_common_process(aubio_process,process_print);

  del_aubio_onset (o);
  del_fvec (onset);

  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  return 0;
}

