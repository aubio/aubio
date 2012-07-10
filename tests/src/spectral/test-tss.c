/* test sample for phase vocoder 
 *
 * this program should start correctly using JACK_START_SERVER=true and
 * reconstruct each audio input frame perfectly on the corresponding input with
 * a delay equal to the window size, hop_s.
 */

#include <stdio.h>
#define AUBIO_UNSTABLE 1
#include <aubio.h>

int main(){
  int i;
  uint_t win_s    = 1024; /* window size                       */
  uint_t hop_s    = 256;  /* hop size                          */

  /* allocate some memory */
  fvec_t * in       = new_fvec (hop_s); /* input buffer       */
  cvec_t * fftgrain = new_cvec (win_s); /* fft norm and phase */
  cvec_t * cstead   = new_cvec (win_s); /* fft norm and phase */
  cvec_t * ctrans   = new_cvec (win_s); /* fft norm and phase */
  fvec_t * stead    = new_fvec (hop_s); /* output buffer      */
  fvec_t * trans    = new_fvec (hop_s); /* output buffer      */
  /* allocate phase vocoders and transient steady-state separation */
  aubio_pvoc_t * pv = new_aubio_pvoc (win_s,hop_s);
  aubio_pvoc_t * pvt = new_aubio_pvoc(win_s,hop_s);
  aubio_pvoc_t * pvs = new_aubio_pvoc(win_s,hop_s);
  aubio_tss_t *  tss = new_aubio_tss(win_s,hop_s);

  /* fill input with some data */
  printf("initialised\n");

  /* execute stft */
  for (i = 0; i < 10; i++) {
    aubio_pvoc_do (pv,in,fftgrain);
    aubio_tss_do  (tss,fftgrain,ctrans,cstead);
    aubio_pvoc_rdo(pvt,cstead,stead);
    aubio_pvoc_rdo(pvs,ctrans,trans);
  }

  del_aubio_pvoc(pv);
  del_aubio_pvoc(pvt);
  del_aubio_pvoc(pvs);
  del_aubio_tss(tss);

  del_fvec(in);
  del_cvec(fftgrain);
  del_cvec(cstead);
  del_cvec(ctrans);
  del_fvec(stead);
  del_fvec(trans);
  aubio_cleanup();
  printf("memory freed\n");
  return 0;
}
