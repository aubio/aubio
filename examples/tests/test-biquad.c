#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        aubio_biquad_t * o = new_aubio_biquad(0.3,0.2,0.1,0.2,0.3);

        aubio_biquad_do_filtfilt(o,in,in);
        aubio_biquad_do(o,in);

        del_aubio_biquad(o);
        del_fvec(in);
        return 0;
}
