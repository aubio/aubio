#include <stdlib.h>
#include "aubio.h"

// Because aubio does not check for double free, this program will crash.
// Programs that call these functions should check for null pointers.

int main (void)
{
  del_fvec(NULL);
  del_lvec(NULL);
  del_cvec(NULL);
  return 0;
}
