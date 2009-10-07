#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t samplerate = 44100;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, 1); /* input buffer */
        aubio_pitchschmitt_t * o  = new_aubio_pitchschmitt(
          win_s, samplerate );
        uint_t i = 0;

        while (i < 1000) {
          aubio_pitchschmitt_do (o,in);
          i++;
        };

        del_aubio_pitchschmitt(o);
        del_fvec(in);
        aubio_cleanup();

        return 0;
}

