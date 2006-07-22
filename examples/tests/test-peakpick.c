#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        aubio_pickpeak_t * o = new_aubio_peakpicker(0.3);

        aubio_peakpick_pimrt(in,o);
        aubio_peakpick_pimrt(in,o);
        aubio_peakpick_pimrt(in,o);
        aubio_peakpick_pimrt(in,o);

        del_aubio_peakpicker(o);
        del_fvec(in);
        return 0;
}

