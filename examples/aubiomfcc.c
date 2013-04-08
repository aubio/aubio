/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>

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

/* mfcc objects */
fvec_t * mfcc_out;
aubio_mfcc_t * mfcc;
aubio_pvoc_t *pv;
cvec_t *fftgrain;

uint_t n_filters = 40;
uint_t n_coefs = 13;

unsigned int pos = 0; /*frames%dspblocksize*/

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
      
      //compute mag spectrum
      aubio_pvoc_do (pv, ibuf, fftgrain);
     
      //compute mfccs
      aubio_mfcc_do(mfcc, fftgrain, mfcc_out);
      
      /* end of block loop */
      pos = -1; /* so it will be zero next j loop */
    }
    pos++;
  }
  return 1;
}

static void process_print (void) {
  /* output times in seconds and extracted mfccs */
  if (sink_uri == NULL) {
    outmsg("%f\t",frames*overlap_size/(float)samplerate);
    fvec_print(mfcc_out);
  }
}

int main(int argc, char **argv) {
  // params
  buffer_size  = 512;
  overlap_size = 256;
  
  examples_common_init(argc,argv);

  /* phase vocoder */
  pv = new_aubio_pvoc (buffer_size, overlap_size);

  fftgrain = new_cvec (buffer_size);

  //populating the filter
  mfcc = new_aubio_mfcc(buffer_size, n_filters, n_coefs, samplerate);
  
  mfcc_out = new_fvec(n_coefs);
  
  //process
  examples_common_process(aubio_process,process_print);
  
  //destroying mfcc 
  del_aubio_pvoc (pv);
  del_cvec (fftgrain);
  del_aubio_mfcc(mfcc);
  del_fvec(mfcc_out);

  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  
  return 0;
}

