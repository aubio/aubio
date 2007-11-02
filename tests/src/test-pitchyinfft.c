#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        aubio_pitchyinfft_t * o  = new_aubio_pitchyinfft(win_s);
        uint_t i = 0;

        while (i < 10) {
          aubio_pitchyinfft_detect (o,in,0.2);
          i++;
        };

        del_aubio_pitchyinfft(o);
        del_fvec(in);
        aubio_cleanup();

        return 0;
}

