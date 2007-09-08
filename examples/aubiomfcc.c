/*
   Copyright (C) 2007 Amaury Hazan

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

/* mfcc objects */
fvec_t * mfcc_outbuf;
aubio_mfcc_t * mfcc;

unsigned int pos = 0; /*frames%dspblocksize*/
uint_t usepitch = 0;

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
      
      //compute mag spectrum
      aubio_pvoc_do (pv,ibuf, fftgrain);
     
      //compute mfccs
      aubio_mfcc_do(mfcc, fftgrain, mfcc_outbuf);

      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

void process_print (void);
void process_print (void) {
      /* output times in seconds
         write extracted mfccs
      */
      
      if (output_filename == NULL) {
        if(frames >= 4) {
          outmsg("%f\n",(frames-4)*overlap_size/(float)samplerate);
        } else if (frames < 4) {
          outmsg("%f\n",0.);
        }
      }
}

int main(int argc, char **argv) {
  // params
  uint_t n_filters = 11;
  smpl_t lowfreq = 500.;
  smpl_t highfreq = 2000.;
  examples_common_init(argc,argv);
  mfcc_outbuf = new_fvec(n_filters,channels);
  
  //populating the filter
  mfcc = new_aubio_mfcc(buffer_size, samplerate, n_filters, lowfreq, highfreq,
      channels);

  //process
  examples_common_process(aubio_process,process_print);
  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  
  //destroying mfcc 
  del_aubio_mfcc(mfcc);
  del_fvec(mfcc_outbuf);
  
  return 0;
}

