#include <aubio.h>
#include <stdlib.h>

int main( int argc, char** argv )
{
  uint_t length;
  for (length = 2; length <= 5; length++)
  {
    fvec_t *t = new_aubio_window(length,aubio_win_rectangle);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_hamming);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_hanning);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_hanningz);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_blackman);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_blackman_harris);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_gaussian);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_welch);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window(length,aubio_win_parzen);
    fvec_print(t);
    del_fvec(t);
  }
  return 0;
}

