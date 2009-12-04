#include <stdio.h>
#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        fvec_t * out      = new_fvec (2);     /* input buffer */
        aubio_tempo_t * o  = new_aubio_tempo("complex", win_s, win_s/4, 44100.);
        uint_t i = 0;

        smpl_t curtempo, curtempoconf;

        while (i < 1000) {
          aubio_tempo_do(o,in,out);
          curtempo = aubio_tempo_get_bpm(o);
          if (curtempo != 0.) {
            fprintf(stdout,"%f\n",curtempo);
            return 1;
          }
          curtempoconf = aubio_tempo_get_confidence(o);
          if (curtempoconf != 0.) {
            fprintf(stdout,"%f\n",curtempo);
            return 1;
          }
          i++;
        };

        del_aubio_tempo(o);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}
