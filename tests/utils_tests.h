#include <stdlib.h>
#include <stdio.h>

#define PRINT_ERR(format, args...)   fprintf(stderr, "AUBIO ERROR: " format , ##args)
#define PRINT_MSG(format, args...)   fprintf(stdout, format , ##args)
#define PRINT_DBG(format, args...)   fprintf(stderr, format , ##args)
#define PRINT_WRN(...)               fprintf(stderr, "AUBIO WARNING: " format, ##args)
