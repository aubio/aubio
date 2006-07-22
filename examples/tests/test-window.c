#include <aubio.h>
#include <stdlib.h>

void print_array(fvec_t *f);
void print_array(fvec_t *f){
  uint i,j;
  for (i=0;i<f->channels;i++)
  {
    for (j=0;j<f->length;j++)
    {
      printf("%1.3e, ", f->data[i][j]); 
    }
    printf(";\n"); 
  }
}

int main( int argc, char** argv )
{
  uint_t length;
  for (length = 2; length <= 5; length++)
  {
    fvec_t *t = new_fvec(length,9);
    aubio_window(t->data[0],t->length,aubio_win_rectangle);
    aubio_window(t->data[1],t->length,aubio_win_hamming);
    aubio_window(t->data[2],t->length,aubio_win_hanning);
    aubio_window(t->data[3],t->length,aubio_win_hanningz);
    aubio_window(t->data[4],t->length,aubio_win_blackman);
    aubio_window(t->data[5],t->length,aubio_win_blackman_harris);
    aubio_window(t->data[6],t->length,aubio_win_gaussian);
    aubio_window(t->data[7],t->length,aubio_win_welch);
    aubio_window(t->data[8],t->length,aubio_win_parzen);
    print_array(t);
    del_fvec(t);
  }
  return 0;
}

