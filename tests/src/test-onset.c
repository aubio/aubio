#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s/4); /* input buffer */
        fvec_t * out      = new_fvec (2);     /* input buffer */
        aubio_onset_t * onset  = new_aubio_onset("complex", win_s, win_s/4, 44100.);
        uint_t i = 0;

        while (i < 10) {
          aubio_onset_do (onset,in,out);
          i++;
        };

        del_aubio_onset(onset);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}
