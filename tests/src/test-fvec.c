#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        fvec_t * in       = new_fvec (win_s); /* input buffer */
        del_fvec(in);

        return 0;
}

