#include <aubio.h>

int main(){
#if HAVE_LIBSAMPLERATE
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        smpl_t ratio      = 0.5;
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec ((uint_t)(win_s*ratio), channels);     /* input buffer */
        aubio_resampler_t * o  = new_aubio_resampler(0.5, 0);
        uint_t i = 0;

        while (i < 100) {
          aubio_resampler_process(o,in,out);
          i++;
        };

        del_aubio_resampler(o);
        del_fvec(in);
        del_fvec(out);

#endif /* HAVE_LIBSAMPLERATE */
        return 0;
}
