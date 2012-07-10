#include <aubio.h>

int main(){
        uint_t length = 1024;                     /* length */
        uint_t height = 1024;                     /* height */
        fmat_t * mat = new_fmat (length, height); /* input buffer */
        fmat_print(mat);
        del_fmat(mat);
        return 0;
}

