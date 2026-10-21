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

#if defined(HAVE_WIN_HACKS) && !defined(__GNUC__)
#include <io.h> // _access
#endif

// This macro is used to pass a string to msvc compiler: since msvc's -D flag
// strips the quotes, we define the string without quotes and re-add them with
// this macro.

#define REDEFINESTRING(x) #x
#define DEFINEDSTRING(x) REDEFINESTRING(x)

#ifndef AUBIO_TESTS_SOURCE
#error "AUBIO_TESTS_SOURCE is not defined"
#endif

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

// Tests that feed aubio bad input on purpose wrap each such call in
// EXPECT_LOGGED: aubio's errors and warnings are counted instead of printed
// while the statement runs, and the test fails if the statement logged
// nothing. The rejection is checked, and the test log keeps only messages
// that point at a real problem.
static uint_t aubio_tests_logged = 0;

static inline void
aubio_tests_count_log (sint_t level, const char_t *message, void *data)
{
  (void)level; (void)message; (void)data;
  aubio_tests_logged++;
}

static inline void
aubio_tests_capture_log (void)
{
  aubio_log_set_level_function (AUBIO_LOG_ERR, aubio_tests_count_log, NULL);
  aubio_log_set_level_function (AUBIO_LOG_WRN, aubio_tests_count_log, NULL);
}

// EXPECT_LOGGED_UNLESS(stmt, ok): for calls whose outcome depends on the
// build (a non-power-of-two FFT fails with ooura, works with fftw): when ok
// is false after stmt, it must have logged why.
#define EXPECT_LOGGED_UNLESS(stmt, ok) do { \
    uint_t logged_before_ = aubio_tests_logged; \
    aubio_tests_capture_log (); \
    stmt; \
    aubio_log_reset (); \
    if (!(ok) && aubio_tests_logged == logged_before_) { \
      PRINT_ERR ("expected an error or warning from: %s\n", #stmt); \
      return 1; \
    } \
  } while (0)

#define EXPECT_LOGGED(stmt) EXPECT_LOGGED_UNLESS(stmt, 0)

#ifndef M_PI
#define M_PI         (3.14159265358979323846)
#endif

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#if defined(HAVE_WIN_HACKS)

// use srand/rand on windows
#define srandom srand
#define random rand

#elif defined(__STRICT_ANSI__)

// workaround to build with -std=c99 (for instance with older cygwin),
// assuming libbc is recent enough to supports these functions.
extern void srandom(unsigned);
extern int random(void);
// mkstemp (int mkstemp(char *)) is POSIX.1-2008, which meson.build selects

#endif

void utils_init_random (void);

void utils_init_random (void) {
  time_t now = time(0);
  struct tm *tm_struct = localtime(&now);
  size_t **tm_address = (void*)&tm_struct;
  unsigned int seed = (unsigned int)(tm_struct->tm_sec + (size_t)tm_address);
  //PRINT_WRN("current seed: %d\n", seed);
  srandom (seed);
}

// create_temp_sink / close_temp_sink
#if defined(__GNUC__) // mkstemp

int check_source(char *source_path)
{
  return access(source_path, R_OK);
}

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

int check_source(char *source_path)
{
  return _access(source_path, 04);
}

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
  const int argc = 2;
  int err = 0;
  char** argv = (char**)calloc(argc, sizeof(char*));
  argv[0] = __FILE__;
  argv[1] = DEFINEDSTRING(AUBIO_TESTS_SOURCE);
  // check if the file can be read
  if ( check_source(argv[1]) ) return 1;
  err = main(argc, argv);
  if (argv) free(argv);
  return err;
}

int run_on_default_sink( int main(int, char**) )
{
  const int argc = 2;
  int err = 0;
  char** argv = (char**)calloc(argc, sizeof(char*));
  char sink_path[PATH_MAX] = "tmp_aubio_XXXXXX";
  int fd = create_temp_sink(sink_path);
  if (!fd) return 1;
  argv[0] = __FILE__;
  argv[1] = sink_path;
  err = main(argc, argv);
  close_temp_sink(sink_path, fd);
  if (argv) free(argv);
  return err;
}

int run_on_default_source_and_sink( int main(int, char**) )
{
  const int argc = 3;
  int err = 0;
  char** argv = (char**)calloc(argc, sizeof(char*));
  char sink_path[PATH_MAX] = "tmp_aubio_XXXXXX";
  int fd = create_temp_sink(sink_path);
  if (!fd) return 1;
  argv[0] = __FILE__;
  argv[1] = DEFINEDSTRING(AUBIO_TESTS_SOURCE);
  argv[2] = sink_path;
  // check if the file can be read
  if ( check_source(argv[1]) ) return 1;
  err = main(argc, argv);
  close_temp_sink(sink_path, fd);
  if (argv) free(argv);
  return err;
}
