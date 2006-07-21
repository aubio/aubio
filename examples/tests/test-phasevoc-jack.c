/* test sample for phase vocoder 
 *
 * this program should start correctly using JACK_START_SERVER=true and
 * reconstruct each audio input frame perfectly on the corresponding input with
 * a delay equal to the window size, hop_s.
 */

#include <unistd.h>  /* pause() or sleep() */
#include <aubio.h>
#include <aubioext.h>

uint_t win_s    = 32; /* window size                       */
uint_t hop_s    = 8;  /* hop size                          */
uint_t channels = 4;  /* number of channels                */
uint_t pos      = 0;  /* frames%dspblocksize for jack loop */
uint_t usejack  = 1;

fvec_t * in;
cvec_t * fftgrain;
fvec_t * out;

aubio_pvoc_t * pv;

aubio_jack_t * jack_setup;

int aubio_process(float **input, float **output, int nframes);

int main(){
        /* allocate some memory */
        in       = new_fvec (hop_s, channels); /* input buffer       */
        fftgrain = new_cvec (win_s, channels); /* fft norm and phase */
        out      = new_fvec (hop_s, channels); /* output buffer      */
        /* allocate fft and other memory space */
        pv = new_aubio_pvoc(win_s,hop_s,channels);
        /* fill input with some data */
        printf("initialised\n");
        /* execute stft */
        aubio_pvoc_do (pv,in,fftgrain);
        printf("computed forward\n");
        /* execute inverse fourier transform */
        aubio_pvoc_rdo(pv,fftgrain,out);
        printf("computed backard\n");

        if (usejack) {
                jack_setup  = new_aubio_jack(channels, channels,
                                (aubio_process_func_t)aubio_process);
                aubio_jack_activate(jack_setup);
                sleep(10); //pause(); /* enter main jack loop */
                aubio_jack_close(jack_setup);
        }
        
        del_aubio_pvoc(pv);
        del_cvec(fftgrain);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();
        printf("memory freed\n");
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
      //for (i=0;i<fftgrain->length;i++) fftgrain->phas[0][i] *= 2.; 
      //for (i=0;i<fftgrain->length;i++) fftgrain->phas[0][i] = 0.; 
      aubio_pvoc_rdo(pv,fftgrain,out);
      pos = -1;
    }
    pos++;
  }
  return 0;
}
