#include <iostream>
#include <aubiocpp.h>

using namespace std;
using namespace aubio;

int main(){
        /* allocate some memory */
        uint_t win_s      = 1024;                       /* window size */
        uint_t channels   = 1;                          /* number of channel */
        cvec c = cvec(win_s, channels); /* input buffer */
        cout << c.norm[0][0] << endl;
        c.norm[0][0] = 2.;
        cout << c.norm[0][0] << endl;
        cout << c.phas[0][0] << endl;
        c.phas[0][0] = 2.;
        cout << c.phas[0][0] << endl;
        return 0;
}


