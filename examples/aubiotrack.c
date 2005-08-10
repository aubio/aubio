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
#include "beattracking.h"

unsigned int pos          = 0;    /* frames%dspblocksize */
unsigned int pos2         = 0;    /* frames%dspblocksize */
uint_t usepitch           = 0;
fvec_t * dfframe          = NULL;
fvec_t * out              = NULL;
aubio_beattracking_t * bt = NULL;
uint_t winlen             = 512;
uint_t step               = 0;
uint_t istactus           = 0;

int aubio_process(float **input, float **output, int nframes);
int aubio_process(float **input, float **output, int nframes) {
  unsigned int i;       /*channels*/
  unsigned int j;       /*frames*/
  smpl_t * btoutput = out->data[0];
  for (j=0;j<nframes;j++) {
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
      /* execute every overlap_size*step */
      if (pos2 == step -1 ) {
              
        /* check dfframe */
        /*
        outmsg("\n");
        for (i = 0; i < winlen; i++ ) 
                outmsg("%f,", dfframe->data[0][i]);
        outmsg("\n");
        */
                        
        aubio_beattracking_do(bt,dfframe,out);

        /* rotate dfframe */
        for (i = 0 ; i < winlen - step; i++ ) 
                dfframe->data[0][i] = dfframe->data[0][i+step];
        for (i = winlen - step ; i < winlen; i++ ) 
                dfframe->data[0][i] = 0.;
                
        pos2 = -1;
      }
      pos2++;
      isonset = aubio_peakpick_pimrt_wt(onset,parms,&(dfframe->data[0][winlen - step + pos2]));
      /* end of second level loop */
      istactus = 0;
      i=0;
      for (i = 1; i < btoutput[0]; i++ ) { 
              if (pos2 == btoutput[i]) {
                      //printf("pos2: %d\n", pos2);
                      //printf("tempo:\t%3.5f bpm \n", 
                      //60.*44100./overlap_size/abs(btoutput[2]-btoutput[1]));
                      /* test for silence */
                      if (aubio_silence_detection(ibuf, threshold2)==1) {
                              isonset  = 0;
                              istactus = 0;
                      } else {
                              istactus = 1;
                      }
              }
      }

      if (istactus ==1) {
              for (pos = 0; pos < overlap_size; pos++)
                      obuf->data[0][pos] = woodblock->data[0][pos];
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
        if (output_filename == NULL) {
                if (istactus)
                        outmsg("%f\n",(frames)*overlap_size/(float)samplerate); 
                if (isonset && verbose)
                        outmsg(" \t \t%f\n",(frames)*overlap_size/(float)samplerate);
        }
}

int main(int argc, char **argv) {
  
  buffer_size = 1024;
  overlap_size = 512;
  /* override default settings */
  examples_common_init(argc,argv);
  winlen = SQR(512)/overlap_size;

  dfframe = new_fvec(winlen,channels);
  step = winlen/4;
  out = new_fvec(step,channels);

  /* test input : impulses starting from 15, at intervals of 50 samples */
  //for(i=0;i<16;i++){ 
  //        dfframe->data[0][50*i+14] = 1.;
  //}

  bt = new_aubio_beattracking(winlen,channels);

  examples_common_process(aubio_process,process_print);

  examples_common_del();

  debug("End of program.\n");

  fflush(stderr);

  return 0;
}

