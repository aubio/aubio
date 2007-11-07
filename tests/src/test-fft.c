
#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 4096;                       /* window size        */
        uint_t channels   = 100;                        /* number of channels */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer       */
        cvec_t * fftgrain = new_cvec (win_s, channels); /* fft norm and phase */
        fvec_t * out      = new_fvec (win_s, channels); /* output buffer      */
        /* allocate fft and other memory space */
        aubio_fft_t * fft = new_aubio_fft(win_s,channels);
        /* fill input with some data */
        //printf("initialised\n");
        /* execute stft */
        aubio_fft_do (fft,in,fftgrain);
        //printf("computed forward\n");
        /* execute inverse fourier transform */
        aubio_fft_rdo(fft,fftgrain,out);
        //printf("computed backard\n");
        del_aubio_fft(fft);
        del_fvec(in);
        del_cvec(fftgrain);
        del_fvec(out);
        //printf("memory freed\n");
        aubio_cleanup();
        return 0;
}
