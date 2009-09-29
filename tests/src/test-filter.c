#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec (win_s, channels);     /* input buffer */
  
        /* allocate fft and other memory space */
        aubio_filter_t * o = new_aubio_filter_c_weighting (44100, channels);

        aubio_filter_do(o,in);
        aubio_filter_do_outplace(o,in,out);
        aubio_filter_do_filtfilt(o,in,out);

        del_aubio_filter(o);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}
