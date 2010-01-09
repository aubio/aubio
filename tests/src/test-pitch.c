#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024; /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        fvec_t * out      = new_fvec (1); /* input buffer */
        aubio_pitch_t *p = new_aubio_pitch ("default", win_s, win_s / 2, 44100);
        uint_t i = 0;

        while (i < 10) {
          aubio_pitch_do (p, in, out);
          i++;
        };

        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}

