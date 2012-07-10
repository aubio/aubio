#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        fvec_t * out      = new_fvec (1); /* output pitch periods */
        aubio_pitchyinfft_t * o  = new_aubio_pitchyinfft(win_s);
        aubio_pitchyinfft_set_tolerance (o, 0.2);
        uint_t i = 0;

        while (i < 10) {
          aubio_pitchyinfft_do (o,in,out);
          i++;
        };

        del_aubio_pitchyinfft(o);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}

