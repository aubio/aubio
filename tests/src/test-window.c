#include <aubio.h>
#include <stdlib.h>

int main( )
{
  uint_t length;
  for (length = 2; length <= 5; length++)
  {
    fvec_t *t = new_aubio_window("rectangle", length);
    del_fvec(t);
    t = new_aubio_window("hamming", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("hanning", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("hanningz", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("blackman", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("blackman_harris", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("gaussian", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("welch", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("parzen", length);
    fvec_print(t);
    del_fvec(t);
    t = new_aubio_window("default", length);
    fvec_print(t);
    del_fvec(t);
  }
  return 0;
}

