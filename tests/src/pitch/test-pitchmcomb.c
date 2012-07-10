#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t hop_s      = win_s/4;                    /* hop size */
        cvec_t * in       = new_cvec (win_s); /* input buffer */
        fvec_t * out      = new_fvec (1); /* input buffer */

        aubio_pitchmcomb_t * o  = new_aubio_pitchmcomb(win_s, hop_s);
        uint_t i = 0;

        while (i < 1000) {
          aubio_pitchmcomb_do (o,in, out);
          i++;
        };

        del_aubio_pitchmcomb(o);
        del_cvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}
