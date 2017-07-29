#include "aubio.h"
#include "utils_tests.h"

int main (void)
{
  aubio_init();

  uint_t vec_size = 10, i;
  fvec_t * vec = new_fvec (vec_size);

  // vec->length matches requested size
  assert(vec->length == vec_size);

  // all elements are initialized to `0.`
  for ( i = 0; i < vec->length; i++ ) {
    assert(vec->data[i] == 0.);
  }

  // all elements can be set to `0.`
  fvec_zeros(vec);
  for ( i = 0; i < vec->length; i++ ) {
    assert(vec->data[i] == 0.);
  }
  fvec_print(vec);

  // all elements can be set to `1.`
  fvec_ones(vec);
  for ( i = 0; i < vec->length; i++ ) {
    assert(vec->data[i] == 1.);
  }
  fvec_print(vec);

  // each element can be accessed directly
  for ( i = 0; i < vec->length; i++ ) {
    vec->data[i] = i;
    assert(vec->data[i] == i);
  }
  fvec_print(vec);

  // now destroys the vector
  del_fvec(vec);

  aubio_cleanup();

  return 0;
}

