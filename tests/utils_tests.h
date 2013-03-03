#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define PRINT_ERR(format, args...)   fprintf(stderr, "AUBIO-TESTS ERROR: " format , ##args)
#define PRINT_MSG(format, args...)   fprintf(stdout, format , ##args)
#define PRINT_DBG(format, args...)   fprintf(stderr, format , ##args)
#define PRINT_WRN(format, args...)   fprintf(stderr, "AUBIO-TESTS WARNING: " format, ##args)

void utils_init_random () {
  time_t now = time(0);
  struct tm *tm_struct = localtime(&now);
  int seed = tm_struct->tm_sec;
  //PRINT_WRN("current seed: %d\n", seed);
  srandom (seed);
}
