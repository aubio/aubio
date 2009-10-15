/*
   Copyright (C) 2003 Paul Brossier

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include "utils.h"

unsigned int pos = 0; /*frames%dspblocksize*/
uint_t usepitch = 0;

int aubio_process(smpl_t **input, smpl_t **output, int nframes);
int aubio_process(smpl_t **input, smpl_t **output, int nframes) {
  unsigned int i;       /*channels*/
  unsigned int j;       /*frames*/
  for (j=0;j<(unsigned)nframes;j++) {
    if(usejack) {
      for (i=0;i<channels;i++) {
        /* write input to datanew */
        fvec_write_sample(ibuf, input[i][j], i, pos);
        /* put synthnew in output */
        output[i][j] = fvec_read_sample(obuf, i, pos);
      }
    }
    /*time for fft*/
    if (pos == overlap_size-1) {         
      /* block loop */
      aubio_pvoc_do (pv,ibuf, fftgrain);
      aubio_onsetdetection_do (o,fftgrain, onset);
      isonset = aubio_peakpicker_do(parms, onset);
      if (isonset) {
        /* test for silence */
        if (aubio_silence_detection(ibuf, silence)==1)
          isonset=0.;
        else
          for (pos = 0; pos < overlap_size; pos++){
            obuf->data[0][pos] = woodblock->data[0][pos];
          }
      } else {
        for (pos = 0; pos < overlap_size; pos++)
          obuf->data[0][pos] = 0.;
      }
      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

void process_print (void);
void process_print (void) {
      /* output times in seconds, taking back some 
       * delay to ensure the label is _before_ the
       * actual onset */
      if (isonset && output_filename == NULL) {
        if(frames >= 4) {
          outmsg("%f\n",(frames - frames_delay + isonset)*overlap_size/(float)samplerate);
        } else if (frames < frames_delay) {
          outmsg("%f\n",0.);
        }
      }
}

int main(int argc, char **argv) {
  frames_delay = 3;
  examples_common_init(argc,argv);
  examples_common_process(aubio_process,process_print);
  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  return 0;
}

