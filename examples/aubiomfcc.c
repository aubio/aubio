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
     
      uint_t coef_cnt;
      uint_t n_filters=20;
      smpl_t outbuf[20];

      for (coef_cnt=0; coef_cnt<n_filters ; coef_cnt++)
        outbuf[coef_cnt]=0.f;
       
      //compute mfccs
      aubio_mffc_do(fftgrain->norm, nframes, mf, outbuf);
      
      for (coef_cnt=0; coef_cnt<n_filters ; coef_cnt++)
        outmsg("%f ",outbuf[coef_cnt]);
      outmsg("\n");
      
      

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
  examples_common_init(argc,argv);
  
  //allocate and initialize mel filter bank
  uint_t n_filters=20;
  uint_t nyquist= samplerate / 2.; 
  smpl_t lowfreq=80.f;
  smpl_t highfreq=18000.f;

  uint_t banksize = (uint) ( sizeof(aubio_mel_filter));
  aubio_mel_filter * mf = (aubio_mel_filter *)getbytes(banksize);

  mf->n_filters = 20;
  mf->filters = (smpl_t **)getbytes(mf->n_filters * sizeof(smpl_t *));
  for(n = 0; n < mf->n_filters; n++)
    mf->filters[n] = (smpl_t *)getbytes((buffer_size/2+1) * sizeof(smpl_t));
  
  //populating the filter
  aubio_mfcc_init(buffer_size, nyquist, XTRACT_EQUAL_GAIN, lowfreq, highfreq, mf->n_filters, mf->filters);

  //process
  examples_common_process(aubio_process,process_print);
  examples_common_del();
  debug("End of program.\n");
  fflush(stderr);
  
  //destroying filterbank
  free(mf);
  
  return 0;
}

