#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        fvec_t * out      = new_fvec (win_s/2); /* input buffer */
        aubio_pitchyin_t *p = new_aubio_pitchyin (win_s);
        uint_t i = 0;

        while (i < 10) {
          aubio_pitchyin_do (p, in,out);
          i++;
        };

        del_fvec(in);
        del_fvec(out);
        del_aubio_pitchyin(p);
        aubio_cleanup();

        return 0;
}
