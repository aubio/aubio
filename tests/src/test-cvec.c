#include <aubio.h>

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        cvec_t * sp       = new_cvec (win_s); /* input buffer */
        del_cvec(sp);

        return 0;
}

