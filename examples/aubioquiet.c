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
sint_t wassilence = 1, issilence;

int aubio_process(smpl_t **input, smpl_t **output, int nframes);
int aubio_process(smpl_t **input, smpl_t **output, int nframes) {
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
      /* test for silence */
      if (aubio_silence_detection(ibuf, silence)==1) {
        if (wassilence==1) issilence = 1;
        else issilence = 2;
        wassilence=1;
      } else { 
        if (wassilence<=0) issilence = 0;
        else issilence = -1;
        wassilence=0;
      }
      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

static void process_print (void) {
      int curframes = (frames - 4) > 0 ? frames -4 : 0;
      if (issilence == -1) {
          outmsg("NOISY: %f\n",curframes*overlap_size/(float)samplerate);
      } else if (issilence == 2) { 
          outmsg("QUIET: %f\n",curframes*overlap_size/(float)samplerate);
      }
}

int main(int argc, char **argv) {
  examples_common_init(argc,argv);
  examples_common_process(aubio_process,process_print);
  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  return 0;
}

