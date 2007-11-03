#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        cvec_t * sp       = new_cvec (win_s, channels); /* input buffer */
        del_cvec(sp);

        return 0;
}

