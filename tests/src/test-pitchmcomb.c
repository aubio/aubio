#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t hop_s      = win_s/4;                    /* hop size */
        uint_t samplerate = 44100;
        uint_t channels   = 1;                          /* number of channel */
        cvec_t * in       = new_cvec (win_s, channels); /* input buffer */
        aubio_pitchmcomb_t * o  = new_aubio_pitchmcomb(
          win_s, hop_s, channels, samplerate );
        uint_t i = 0;

        while (i < 1000) {
          aubio_pitchmcomb_do (o,in);
          i++;
        };

        del_aubio_pitchmcomb(o);
        del_cvec(in);
        aubio_cleanup();

        return 0;
}
