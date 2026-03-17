#include <aubio.h>
#include "utils_tests.h"
#include <time.h>

#define N_FRAMES 100000
#define WIN_S    1024
#define HOP_S    256

int main (void)
{
  uint_t i;
  fvec_t *in       = new_fvec(HOP_S);
  cvec_t *fftgrain = new_cvec(WIN_S);
  aubio_pvoc_t *pv = new_aubio_pvoc(WIN_S, HOP_S);
  if (!in || !fftgrain || !pv) return 1;

  /* fill with arbitrary non-zero data */
  for (i = 0; i < HOP_S; i++) in->data[i] = (smpl_t)i / HOP_S;

  clock_t t0 = clock();
  for (i = 0; i < N_FRAMES; i++)
    aubio_pvoc_do(pv, in, fftgrain);
  double ms_full = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

  t0 = clock();
  for (i = 0; i < N_FRAMES; i++)
    aubio_pvoc_do_norm(pv, in, fftgrain);
  double ms_norm = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

  PRINT_MSG("pvoc_do      (full):      %.1f ms for %d frames (%.2f us/frame)\n",
      ms_full, N_FRAMES, ms_full * 1000.0 / N_FRAMES);
  PRINT_MSG("pvoc_do_norm (no phase):  %.1f ms for %d frames (%.2f us/frame)\n",
      ms_norm, N_FRAMES, ms_norm * 1000.0 / N_FRAMES);
  PRINT_MSG("speedup: %.2fx\n", ms_full / ms_norm);

  del_aubio_pvoc(pv);
  del_fvec(in);
  del_cvec(fftgrain);
  aubio_cleanup();
  return 0;
}
