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

#include <aubio.h>
#include "utils.h"

uint_t pos = 0;    /* frames%dspblocksize */
fvec_t * tempo_out = NULL;
aubio_tempo_t * bt = NULL;
smpl_t istactus = 0;
smpl_t isonset = 0;

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
      aubio_tempo_do (bt,ibuf,tempo_out);
      istactus = fvec_read_sample (tempo_out, 0);
      isonset = fvec_read_sample (tempo_out, 1);
      if (istactus > 0.) {
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

static void process_print (void) {
        if (sink_uri == NULL) {
                if (istactus) {
                        outmsg("%f\n",((smpl_t)(frames*overlap_size)+(istactus-1.)*overlap_size)/(smpl_t)samplerate); 
                }
                if (isonset && verbose)
                        outmsg(" \t \t%f\n",(frames)*overlap_size/(float)samplerate);
        }
}

int main(int argc, char **argv) {
  
  buffer_size = 1024;
  overlap_size = 512;
  /* override default settings */
  examples_common_init(argc,argv);

  tempo_out = new_fvec(2);
  bt = new_aubio_tempo(onset_mode,buffer_size,overlap_size, samplerate);
  if (threshold != 0.) aubio_tempo_set_threshold (bt, threshold);

  examples_common_process(aubio_process,process_print);

  del_aubio_tempo(bt);
  del_fvec(tempo_out);

  examples_common_del();

  debug("End of program.\n");

  fflush(stderr);

  return 0;
}

