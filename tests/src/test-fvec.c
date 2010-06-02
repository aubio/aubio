#include <aubio.h>
#include <assert.h>

int main(){
  uint_t buffer_size = 1024;
  fvec_t * in = new_fvec (buffer_size);

  assert( in->length                == buffer_size);

  assert( in->data[0]               == 0);
  assert( in->data[buffer_size / 2] == 0);
  assert( in->data[buffer_size - 1] == 0);

  in->data[buffer_size -1 ] = 1;
  assert( in->data[buffer_size - 1] == 1);

  del_fvec(in);

  return 0;
}

