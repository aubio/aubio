#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "config.h"

#define PRINT_ERR(format, args...)   fprintf(stderr, "AUBIO-TESTS ERROR: " format , ##args)
#define PRINT_MSG(format, args...)   fprintf(stdout, format , ##args)
#define PRINT_DBG(format, args...)   fprintf(stderr, format , ##args)
#define PRINT_WRN(format, args...)   fprintf(stderr, "AUBIO-TESTS WARNING: " format, ##args)

#ifdef HAVE_WIN_HACKS
// http://en.wikipedia.org/wiki/Linear_congruential_generator
// no srandom/random on win32

uint_t srandom_seed = 1029;

void srandom(uint_t new_seed) {
    srandom_seed = new_seed;
}

uint_t random(void) {
    srandom_seed = 1664525 * srandom_seed + 1013904223;
    return srandom_seed;
}
#endif

void utils_init_random (void);

void utils_init_random (void) {
  time_t now = time(0);
  struct tm *tm_struct = localtime(&now);
  int seed = tm_struct->tm_sec;
  //PRINT_WRN("current seed: %d\n", seed);
  srandom (seed);
}
