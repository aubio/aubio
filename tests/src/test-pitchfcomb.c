#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t hop_s      = win_s/4;                    /* hop size */
        uint_t channels   = 3;                          /* number of channel */
        fvec_t * in       = new_fvec (hop_s, channels); /* input buffer */
        fvec_t * out      = new_fvec (1, channels);
        aubio_pitchfcomb_t * o  = new_aubio_pitchfcomb (
          win_s, hop_s, channels);
        uint_t i = 0;

        while (i < 2) {
          aubio_pitchfcomb_do (o,in, out);
          i++;
        };

        del_aubio_pitchfcomb(o);
        del_fvec(in);
        aubio_cleanup();

        return 0;
}
