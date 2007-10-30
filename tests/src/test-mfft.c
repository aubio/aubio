
#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 4096;                       /* window size        */
        uint_t channels   = 100;                        /* number of channels */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer       */
        cvec_t * fftgrain = new_cvec (win_s, channels); /* fft norm and phase */
        fvec_t * out      = new_fvec (win_s, channels); /* output buffer      */
        /* allocate fft and other memory space */
        aubio_mfft_t * fft = new_aubio_mfft(win_s,channels);
        /* fill input with some data */
        //printf("initialised\n");
        /* execute stft */
        aubio_mfft_do (fft,in,fftgrain);
        //printf("computed forward\n");
        /* execute inverse fourier transform */
        aubio_mfft_rdo(fft,fftgrain,out);
        //printf("computed backard\n");
        del_aubio_mfft(fft);
        del_fvec(in);
        del_cvec(fftgrain);
        del_fvec(out);
        //printf("memory freed\n");
        aubio_cleanup();
        return 0;
}
