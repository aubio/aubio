#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec_t * in       = new_fvec (win_s, channels); /* input buffer */
        fvec_t * out      = new_fvec (win_s/4, channels);     /* input buffer */
  
        /* allocate fft and other memory space */
        aubio_beattracking_t * tempo  = new_aubio_beattracking(win_s, channels);

        uint_t i = 0;

        smpl_t curtempo, curtempoconf;

        while (i < 10) {
          aubio_beattracking_do(tempo,in,out);
          curtempo = aubio_beattracking_get_bpm(tempo);
          if (curtempo != 0.) {
            fprintf(stdout,"%f\n",curtempo);
            return 1;
          }
          curtempoconf = aubio_beattracking_get_confidence(tempo);
          if (curtempoconf != 0.) {
            fprintf(stdout,"%f\n",curtempo);
            return 1;
          }
          i++;
        };

        del_aubio_beattracking(tempo);
        del_fvec(in);
        del_fvec(out);
        aubio_cleanup();

        return 0;
}

