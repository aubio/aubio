#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec (1, channels); /* input buffer */
        aubio_peakpicker_t * o = new_aubio_peakpicker(1);
        aubio_peakpicker_set_threshold (o, 0.3);

        aubio_peakpicker_do(o, in, out);
        aubio_peakpicker_do(o, in, out);
        aubio_peakpicker_do(o, in, out);
        aubio_peakpicker_do(o, in, out);

        del_aubio_peakpicker(o);
        del_fvec(in);
        return 0;
}

