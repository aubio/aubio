/** Test for phase vocoder in jack
 *
 * This program should start correctly, when jackd is started or when
 * using JACK_START_SERVER=true and reconstruct each audio input channel
 * on the corresponding output channel with some strange effects and a
 * delay equal to the hop size (hop_s).
 *
 */

#include <stdio.h>
#include <unistd.h>  /* sleep() */
#include <aubio.h>
#ifdef HAVE_JACK
#include "jackio.h"
#endif /* HAVE_JACK */

uint_t testing  = 0;  /* change this to 1 to listen        */

uint_t win_s    = 512;/* window size                       */
uint_t hop_s    = 128;/* hop size                          */
uint_t channels = 2;  /* number of audio channels          */
uint_t midiin   = 4;  /* number of midi input channels     */
uint_t midiout  = 2;  /* number of midi output channels    */
uint_t pos      = 0;  /* frames%dspblocksize for jack loop */

fvec_t * in[2];
cvec_t * fftgrain[2];
fvec_t * out[2];

aubio_pvoc_t * pv[2];

int aubio_process(float **input, float **output, int nframes);

int main(){
        /* allocate some memory */
  uint_t i;
    for (i=0;i<channels;i++) {
        in[i]       = new_fvec (hop_s); /* input buffer       */
        fftgrain[i] = new_cvec (win_s); /* fft norm and phase */
        out[i]      = new_fvec (hop_s); /* output buffer      */
        /* allocate fft and other memory space */
        pv[i] = new_aubio_pvoc(win_s,hop_s);
    }

#ifdef HAVE_JACK
        aubio_jack_t * jack_setup;
        jack_setup  = new_aubio_jack(channels, channels, 
            midiin, midiout,
            (aubio_process_func_t)aubio_process);
        aubio_jack_activate(jack_setup);
        /* stay in main jack loop for 1 seconds only */
        do {
          sleep(1);
        } while(testing);
        aubio_jack_close(jack_setup);
#else
        fprintf(stderr, "WARNING: no jack support\n");
#endif
        
    for (i=0;i<channels;i++) {
        del_aubio_pvoc(pv[i]);
        del_cvec(fftgrain[i]);
        del_fvec(in[i]);
        del_fvec(out[i]);
    }
        aubio_cleanup();
        return 0;
}

int aubio_process(float **input, float **output, int nframes) {
  uint_t i;       /*channels*/
  uint_t j;       /*frames*/
  for (j=0;j<(unsigned)nframes;j++) {
    for (i=0;i<channels;i++) {
      /* write input to datanew */
      fvec_write_sample(in[i], input[i][j], pos);
      /* put synthnew in output */
      output[i][j] = fvec_read_sample(out[i], pos);
    }
    /*time for fft*/
    if (pos == hop_s-1) {
      /* block loop */
    for (i=0;i<channels;i++) {
      aubio_pvoc_do (pv[i], in[i], fftgrain[i]);
      // zero phases of first channel
      for (i=0;i<fftgrain[i]->length;i++) fftgrain[0]->phas[i] = 0.; 
      // double phases of second channel
      for (i=0;i<fftgrain[i]->length;i++) {
        fftgrain[1]->phas[i] = 
          aubio_unwrap2pi (fftgrain[1]->phas[i] * 2.); 
      }
      // copy second channel to third one
      aubio_pvoc_rdo(pv[i], fftgrain[i], out[i]);
      pos = -1;
    }
    }
    pos++;
  }
  return 0;
}
