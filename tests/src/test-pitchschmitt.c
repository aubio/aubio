#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        fvec_t * out = new_fvec (1); /* input buffer */
        aubio_pitchschmitt_t * o  = new_aubio_pitchschmitt(win_s);
        uint_t i = 0;

        while (i < 1000) {
          aubio_pitchschmitt_do (o,in, out);
          i++;
        };

        del_aubio_pitchschmitt(o);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}

