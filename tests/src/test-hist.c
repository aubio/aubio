#include <aubio.h>
#include <stdlib.h>

void print_array(fvec_t *f);
void print_array(fvec_t *f){
  uint_t i,j;
  for (i=0;i<f->channels;i++){
    for (j=0;j<f->length;j++){
      printf("%f, ", f->data[i][j]); 
    }
    printf(";\n"); 
  }
}

int main( int argc, char** argv )
{
  uint_t length;
  for (length = 1; length < 10; length ++ ) {
    fvec_t *t = new_fvec(length,5);
    aubio_hist_t *o = new_aubio_hist(0, 1, length, 5);
    aubio_window(t->data[0],t->length,aubio_win_hanning);
    aubio_window(t->data[1],t->length,aubio_win_hanningz);
    aubio_window(t->data[2],t->length,aubio_win_blackman);
    aubio_window(t->data[3],t->length,aubio_win_blackman_harris);
    aubio_window(t->data[4],t->length,aubio_win_hamming);
    print_array(t);
    aubio_hist_do(o,t);
    print_array(t);
    aubio_hist_do_notnull(o,t);
    print_array(t);
    aubio_hist_dyn_notnull(o,t);
    print_array(t);
    del_aubio_hist(o);
    del_fvec(t);
  }
  return 0;
}

