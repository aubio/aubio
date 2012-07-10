#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        aubio_filter_t * o = new_aubio_filter_biquad(0.3,0.2,0.1,0.2,0.3);

        aubio_filter_do_filtfilt(o,in,in);
        aubio_filter_do(o,in);

        del_aubio_filter(o);
        del_fvec(in);
        return 0;
}
