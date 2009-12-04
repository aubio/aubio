#define AUBIO_UNSTABLE 1

#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        aubio_scale_t * o = new_aubio_scale(0,1,2,3);
        aubio_scale_set_limits (o,0,1,2,3);
        uint_t i = 0;

        while (i < 1000) {
          aubio_scale_do(o,in);
          i++;
        };

        del_aubio_scale(o);
        del_fvec(in);

        return 0;
}
