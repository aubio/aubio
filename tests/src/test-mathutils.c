#include <stdio.h>
#include <assert.h>
#include <aubio.h>

int main(){
  int a, b;

  a = 31; b = aubio_next_power_of_two(a);
  fprintf(stdout, "next_power_of_two of %d is %d\n", a, b);
  assert(b == 32);

  a = 32; b = aubio_next_power_of_two(a);
  fprintf(stdout, "next_power_of_two of %d is %d\n", a, b);
  assert(b == 32);

  a = 33; b = aubio_next_power_of_two(a);
  fprintf(stdout, "next_power_of_two of %d is %d\n", a, b);
  assert(b == 64);

  return 0;
}

