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
uint_t usepitch = 1;

int aubio_process(float **input, float **output, int nframes);
int aubio_process(float **input, float **output, int nframes) {
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
      aubio_onsetdetection(o,fftgrain, onset);
      if (usedoubled) {
        aubio_onsetdetection(o2,fftgrain, onset2);
        onset->data[0][0] *= onset2->data[0][0];
      }
      isonset = aubio_peakpick_pimrt(onset,parms);
      
      pitch = aubio_pitchdetection(pitchdet,ibuf);
      if(median){
              note_append(note_buffer, pitch);
      }

      /* curlevel is negatif or 1 if silence */
      curlevel = aubio_level_detection(ibuf, silence);
      if (isonset) {
              /* test for silence */
              if (curlevel == 1.) {
                      isonset=0;
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
                              obuf->data[0][pos] = woodblock->data[0][pos];
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
      if (verbose) outmsg("%f\n",pitch);
}

int main(int argc, char **argv) {
  examples_common_init(argc,argv);
  examples_common_process(aubio_process, process_print);
  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  return 0;
}

