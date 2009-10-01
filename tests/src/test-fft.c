
#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 8;                       /* window size        */
        uint_t channels   = 1;                        /* number of channels */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer       */
        cvec_t * fftgrain = new_cvec (win_s, channels); /* fft norm and phase */
        fvec_t * out      = new_fvec (win_s, channels); /* output buffer      */
        in->data[0][0] = 1;
        in->data[0][1] = 2;
        in->data[0][2] = 3;
        in->data[0][3] = 4;
        in->data[0][4] = 5;
        in->data[0][5] = 6;
        in->data[0][6] = 5;
        in->data[0][7] = 6;
        /* allocate fft and other memory space */
        aubio_fft_t * fft = new_aubio_fft(win_s,channels);
        /* fill input with some data */
        fvec_print(in);
        /* execute stft */
        aubio_fft_do (fft,in,fftgrain);
        cvec_print(fftgrain);
        /* execute inverse fourier transform */
        aubio_fft_rdo(fft,fftgrain,out);
        fvec_print(out);
        del_aubio_fft(fft);
        del_fvec(in);
        del_cvec(fftgrain);
        del_fvec(out);
        aubio_cleanup();
        return 0;
}
