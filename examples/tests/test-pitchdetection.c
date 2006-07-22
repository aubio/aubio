#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t hop_s      = win_s/4;                    /* hop size */
        uint_t samplerate = 44100;                      /* samplerate */
        uint_t channels   = 1;                          /* number of channel */
        aubio_pitchdetection_mode mode = aubio_pitchm_freq;
        aubio_pitchdetection_type type = aubio_pitch_yinfft;
        fvec_t * in       = new_fvec (hop_s, channels); /* input buffer */
        aubio_pitchdetection_t * o  = new_aubio_pitchdetection(
          win_s, hop_s, channels, samplerate, type, mode
          );
        uint_t i = 0;

        while (i < 1000) {
          aubio_pitchdetection(o,in);
          i++;
        };

        del_aubio_pitchdetection(o);
        del_fvec(in);
        aubio_cleanup();

        return 0;
}
