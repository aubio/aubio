#include <aubio.h>

int main()
{
  uint_t win_s = 1024; // window size
  lvec_t * sp = new_lvec (win_s); // input buffer
  del_lvec(sp);
  return 0;
}

