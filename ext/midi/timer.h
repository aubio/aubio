
#if defined(WIN32)
#define aubio_curtime()   GetTickCount()

double aubio_utime(void);

#elif defined(MACOS9)
#include <OSUtils.h>
#include <Timer.h>

unsigned int aubio_curtime();
#define aubio_utime()  0.0

#else

unsigned int aubio_curtime(void);
double aubio_utime(void);

#endif

/* if the callback function returns 1 the timer will continue; if it
   returns 0 it will stop */
typedef int (*aubio_timer_callback_t)(void* data, unsigned int msec);

typedef struct _aubio_timer_t aubio_timer_t;

aubio_timer_t* new_aubio_timer(int msec, aubio_timer_callback_t callback, 
                        void* data, int new_thread, int auto_destroy);

int delete_aubio_timer(aubio_timer_t* timer);
int aubio_timer_join(aubio_timer_t* timer);
int aubio_timer_stop(aubio_timer_t* timer);
void * aubio_timer_start(void * data);
void aubio_time_config(void);
