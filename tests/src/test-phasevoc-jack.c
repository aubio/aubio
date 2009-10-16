/** Test for phase vocoder in jack
 *
 * This program should start correctly, when jackd is started or when
 * using JACK_START_SERVER=true and reconstruct each audio input channel
 * on the corresponding output channel with some strange effects and a
 * delay equal to the hop size (hop_s).
 *
 */

#include <unistd.h>  /* sleep() */
#include <aubio.h>
#include "jackio.h"

uint_t testing  = 1;  /* change this to 1 to listen        */

uint_t win_s    = 512;/* window size                       */
uint_t hop_s    = 128;/* hop size                          */
uint_t channels = 2;  /* number of audio channels          */
uint_t midiin   = 4;  /* number of midi input channels     */
uint_t midiout  = 2;  /* number of midi output channels    */
uint_t pos      = 0;  /* frames%dspblocksize for jack loop */

fvec_t * in;
cvec_t * fftgrain;
fvec_t * out;

aubio_pvoc_t * pv;

int aubio_process(float **input, float **output, int nframes);

int main(){
        /* allocate some memory */
        in       = new_fvec (hop_s, channels); /* input buffer       */
        fftgrain = new_cvec (win_s, channels); /* fft norm and phase */
        out      = new_fvec (hop_s, channels); /* output buffer      */
        /* allocate fft and other memory space */
        pv = new_aubio_pvoc(win_s,hop_s,channels);

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
#endif
        
        del_aubio_pvoc(pv);
        del_cvec(fftgrain);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();
        return 0;
}

int aubio_process(float **input, float **output, int nframes) {
  uint_t i;       /*channels*/
  uint_t j;       /*frames*/
  for (j=0;j<(unsigned)nframes;j++) {
    for (i=0;i<channels;i++) {
      /* write input to datanew */
      fvec_write_sample(in, input[i][j], i, pos);
      /* put synthnew in output */
      output[i][j] = fvec_read_sample(out, i, pos);
    }
    /*time for fft*/
    if (pos == hop_s-1) {
      /* block loop */
      aubio_pvoc_do (pv,in, fftgrain);
      // zero phases of first channel
      for (i=0;i<fftgrain->length;i++) fftgrain->phas[0][i] = 0.; 
      // double phases of second channel
      for (i=0;i<fftgrain->length;i++) {
        fftgrain->phas[1][i] = 
          aubio_unwrap2pi (fftgrain->phas[1][i] * 2.); 
      }
      // copy second channel to third one
      aubio_pvoc_rdo(pv,fftgrain,out);
      pos = -1;
    }
    pos++;
  }
  return 0;
}
