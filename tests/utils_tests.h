#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h> // unlink, close
#endif

#ifdef HAVE_LIMITS_H
#include <limits.h> // PATH_MAX
#endif /* HAVE_LIMITS_H */
#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

#define DEFAULT_TEST_FILE "python/tests/sounds/44100Hz_44100f_sine441.wav"

#ifdef HAVE_C99_VARARGS_MACROS
#define PRINT_ERR(...)   fprintf(stderr, "AUBIO-TESTS ERROR: " __VA_ARGS__)
#define PRINT_MSG(...)   fprintf(stdout, __VA_ARGS__)
#define PRINT_DBG(...)   fprintf(stderr, __VA_ARGS__)
#define PRINT_WRN(...)   fprintf(stderr, "AUBIO-TESTS WARNING: " __VA_ARGS__)
#else
#define PRINT_ERR(format, args...)   fprintf(stderr, "AUBIO-TESTS ERROR: " format , ##args)
#define PRINT_MSG(format, args...)   fprintf(stdout, format , ##args)
#define PRINT_DBG(format, args...)   fprintf(stderr, format , ##args)
#define PRINT_WRN(format, args...)   fprintf(stderr, "AUBIO-TESTS WARNING: " format, ##args)
#endif

#ifndef M_PI
#define M_PI         (3.14159265358979323846)
#endif

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

// are we on windows ? or are we using -std=c99 ?
#if defined(HAVE_WIN_HACKS) || defined(__STRICT_ANSI__)
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
  size_t **tm_address = (void*)&tm_struct;
  int seed = tm_struct->tm_sec + (size_t)tm_address;
  //PRINT_WRN("current seed: %d\n", seed);
  srandom ((unsigned int)seed);
}

// create_temp_sink / close_temp_sink
#if defined(__GNUC__) // mkstemp

int create_temp_sink(char *sink_path)
{
  return mkstemp(sink_path);
}

int close_temp_sink(char *sink_path, int sink_fildes)
{
  int err;
  if ((err = close(sink_fildes)) != 0) return err;
  if ((err = unlink(sink_path)) != 0) return err;
  return err;
}

#elif defined(HAVE_WIN_HACKS) //&& !defined(__GNUC__)
// windows workaround, where mkstemp does not exist...
int create_temp_sink(char *templ)
{
  int i = 0;
  static const char letters[] = "abcdefg0123456789";
  int letters_len = strlen(letters);
  int templ_len = strlen(templ);
  if (templ_len == 0) return 0;
  utils_init_random();
  for (i = 0; i < 6; i++)
  {
    templ[templ_len - i] = letters[rand() % letters_len];
  }
  return 1;
}

int close_temp_sink(char* sink_path, int sink_fildes) {
  // the file should be closed when not using mkstemp, no need to open it
  if (sink_fildes == 0) return 1;
  return _unlink(sink_path);
}

#else // windows workaround
// otherwise, we don't really know what to do yet
#error "mkstemp undefined, but not on windows. additional workaround required."
#endif

// pass progname / default
int run_on_default_source( int main(int, char**) )
{
  int argc = 2;
  char* argv[argc];
  argv[0] = __FILE__;
  // when running from waf build
  argv[1] = "../../" DEFAULT_TEST_FILE;
  // when running from source root directory
  if ( access(argv[1], R_OK) )
      argv[1] = DEFAULT_TEST_FILE;
  // no file found
  if ( access(argv[1], R_OK) != 0 )
      return 1;
  return main(argc, argv);
}

int run_on_default_source_and_sink( int main(int, char**) )
{
  int argc = 3, err;
  char* argv[argc];
  argv[0] = __FILE__;
  // when running from waf build
  argv[1] = "../../" DEFAULT_TEST_FILE;
  // when running from source root directory
  if ( access(argv[1], R_OK) )
      argv[1] = DEFAULT_TEST_FILE;
  // no file found
  if ( access(argv[1], R_OK) != 0 )
      return 1;
  char sink_path[PATH_MAX] = "tmp_aubio_XXXXXX";
  int fd = mkstemp(sink_path);
  if (!fd) return 1;
  argv[2] = sink_path;
  err = main(argc, argv);
  unlink(sink_path);
  close(fd);
  return err;
}

int run_on_default_sink( int main(int, char**) )
{
  const int argc = 2;
  int err = 0;
  char* argv[argc];
  char sink_path[PATH_MAX] = "tmp_aubio_XXXXXX";
  int fd = create_temp_sink(sink_path);
  if (!fd) return 1;
  argv[0] = __FILE__;
  argv[1] = sink_path;
  err = main(argc, argv);
  close_temp_sink(sink_path, fd);
  return err;
}
