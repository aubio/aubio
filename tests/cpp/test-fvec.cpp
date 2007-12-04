#include <iostream>
#include <aubiocpp.h>

using namespace std;
using namespace aubio;

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        fvec f = fvec(win_s, channels); /* input buffer */
        cout << f[0][0] << endl;
        f[0][0] = 2.;
        cout << f[0][0] << endl;
        return 0;
}

