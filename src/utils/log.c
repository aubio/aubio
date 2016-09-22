#include "config.h"
#include "aubio_priv.h"
#include "log.h"

/** array of pointers to logging functions, one per level */
static aubio_log_function_t aubio_log_function[AUBIO_LOG_LAST_LEVEL];
/** array of pointers to closure passed to logging functions, one per level */
static void* aubio_log_user_data[AUBIO_LOG_LAST_LEVEL];
/** buffer for logging messages */
static char aubio_log_buffer[512];

/** private function used by default by logging functions */
void
aubio_default_log(sint_t level, const char_t *message, void * data UNUSED)
{
  FILE *out;
  out = stdout;
  if (level == AUBIO_LOG_DBG || level == AUBIO_LOG_ERR) {
    out = stderr;
  }
  if (data != NULL) {
    fprintf(out, "%s", (char_t *)data);
  }
  fprintf(out, "%s", message);
  //fflush(out);
}

uint_t
aubio_log(sint_t level, const char_t *fmt, ...)
{
  aubio_log_function_t fun = NULL;

  va_list args;
  va_start(args, fmt);
  vsnprintf(aubio_log_buffer, sizeof(aubio_log_buffer), fmt, args);
  va_end(args);

  if ((level >= 0) && (level < AUBIO_LOG_LAST_LEVEL)) {
    fun = aubio_log_function[level];
    if (fun != NULL) {
      (*fun)(level, aubio_log_buffer, aubio_log_user_data[level]);
    } else {
      aubio_default_log(level, aubio_log_buffer, NULL);
    }
  }
  return AUBIO_FAIL;
}

void
aubio_log_reset(void)
{
  uint_t i = 0;
  for (i = 0; i < AUBIO_LOG_LAST_LEVEL; i++) {
    aubio_log_set_level_function(i, aubio_default_log, NULL);
  }
}

aubio_log_function_t
aubio_log_set_level_function(sint_t level, aubio_log_function_t fun, void * data)
{
  aubio_log_function_t old = NULL;
  if ((level >= 0) && (level < AUBIO_LOG_LAST_LEVEL)) {
    old = aubio_log_function[level];
    aubio_log_function[level] = fun;
    aubio_log_user_data[level] = data;
  }
  return old;
}

void
aubio_log_set_function(aubio_log_function_t fun, void * data) {
  uint_t i = 0;
  for (i = 0; i < AUBIO_LOG_LAST_LEVEL; i++) {
    aubio_log_set_level_function(i, fun, data);
  }
}
