#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec (win_s/2, channels); /* input buffer */
        aubio_pitchyin_t *p = new_aubio_pitchyin (win_s);
        uint_t i = 0;

        while (i < 10) {
          aubio_pitchyin_do (p, in,out);
          i++;
        };

        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}
