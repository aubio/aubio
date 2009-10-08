#include <stdio.h>
#include <aubio.h>

int main(){
#if HAVE_SAMPLERATE
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        smpl_t ratio      = 0.5;
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec ((uint_t)(win_s*ratio), channels);     /* input buffer */
        aubio_resampler_t * o  = new_aubio_resampler(0.5, 0);
        uint_t i = 0;

        while (i < 10) {
          aubio_resampler_do(o,in,out);
          i++;
        };

        del_aubio_resampler(o);
        del_fvec(in);
        del_fvec(out);

#else
        fprintf(stderr, "aubio_resampler_t not compiled in\n");
#endif /* HAVE_SAMPLERATE */
        return 0;
}
