#include "aubio.h"
#include "utils_tests.h"

// create a new matrix and fill it with i * 1. + j * .1, where i is the row,
// and j the column.

int main (void)
{
  aubio_init();

  uint_t height = 3, length = 9, i, j;
  // create fmat_t object
  fmat_t * mat = new_fmat (height, length);
  for ( i = 0; i < mat->height; i++ ) {
    for ( j = 0; j < mat->length; j++ ) {
      // all elements are already initialized to 0.
      assert(mat->data[i][j] == 0);
      // setting element of row i, column j
      mat->data[i][j] = i * 1. + j *.1;
    }
  }
  fvec_t channel_onstack;
  fvec_t *channel = &channel_onstack;
  fmat_get_channel(mat, 1, channel);
  fvec_print (channel);
  // print out matrix
  fmat_print(mat);
  // destroy it
  del_fmat(mat);

  aubio_cleanup();
  
  return 0;
}

