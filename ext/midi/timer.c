/* 
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public License
 * as published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *  
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307, USA
 */

/* this file originally taken from FluidSynth - A Software Synthesizer
 * Copyright (C) 2003  Peter Hanappe and others.
 */

#include "aubio_priv.h"
#include "timer.h"

#if defined(WIN32)

/*=============================================================*/
/*                                                             */
/*                           Win32                             */
/*                                                             */
/*=============================================================*/

/***************************************************************
 *
 *               Timer
 *
 */
#include <windef.h>

#if 0
#include <winbase.h>
#endif

struct _aubio_timer_t 
{
  long msec;
  aubio_timer_callback_t callback;
  void* data;
  HANDLE thread;
  DWORD thread_id;
  int cont;
  int auto_destroy;
};

static int aubio_timer_count = 0;
DWORD WINAPI aubio_timer_run(LPVOID data);

aubio_timer_t* 
new_aubio_timer(int msec, aubio_timer_callback_t callback, void* data, 
           int new_thread, int auto_destroy)
{
  aubio_timer_t* timer = AUBIO_NEW(aubio_timer_t);
  if (timer == NULL) {
    AUBIO_ERR( "Out of memory");     
    return NULL;
  }

  timer->cont = 1;
  timer->msec = msec;
  timer->callback = callback;
  timer->data = data;
  timer->thread = 0;
  timer->auto_destroy = auto_destroy;

  if (new_thread) {
#if 0
    timer->thread = CreateThread(NULL, 0, aubio_timer_run, (LPVOID) timer, 0, &timer->thread_id);
#endif
    if (timer->thread == NULL) {
      AUBIO_ERR( "Couldn't create timer thread");     
      AUBIO_FREE(timer);
      return NULL;
    }
#if 0
    SetThreadPriority(timer->thread, THREAD_PRIORITY_TIME_CRITICAL);
#endif
  } else {
    aubio_timer_run((LPVOID) timer); 
  }
  return timer;
}

DWORD WINAPI 
aubio_timer_run(LPVOID data)
{
#if 0
  int count = 0;
  int cont = 1;
  long start;
  long delay;
  aubio_timer_t* timer;
  timer = (aubio_timer_t*) data;

  if ((timer == NULL) || (timer->callback == NULL)) {
    return 0;
  }

  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

  /* keep track of the start time for absolute positioning */
  start = aubio_curtime();

  while (cont) {

    /* do whatever we have to do */
    cont = (*timer->callback)(timer->data, aubio_curtime() - start);

    count++;

    /* to avoid incremental time errors, I calculate the delay between
       two callbacks bringing in the "absolute" time (count *
       timer->msec) */
    delay = (count * timer->msec) - (aubio_curtime() - start);
    if (delay > 0) {
      Sleep(delay);
    }

    cont &= timer->cont;
  }

  AUBIO_DBG( "Timer thread finished");

  if (timer->auto_destroy) {
    AUBIO_FREE(timer);
  }

  ExitThread(0);
#endif
  return 0;
}

int 
delete_aubio_timer(aubio_timer_t* timer)
{
  timer->cont = 0;
  aubio_timer_join(timer); 
  AUBIO_FREE(timer);
  return AUBIO_OK;
}

int 
aubio_timer_join(aubio_timer_t* timer)
{
#if 0
  DWORD wait_result;
  if (timer->thread == 0) {
    return AUBIO_OK;
  }
  wait_result = WaitForSingleObject(timer->thread, INFINITE);
  return (wait_result == WAIT_OBJECT_0)? AUBIO_OK : AUBIO_FAIL;
#else
  return 0;
#endif
}
/***************************************************************
 *
 *               Time
 */

double rdtsc(void);
double aubio_estimate_cpu_frequency(void);

static double aubio_cpu_frequency = -1.0;

void aubio_time_config(void)
{
  if (aubio_cpu_frequency < 0.0) {
    aubio_cpu_frequency = aubio_estimate_cpu_frequency() / 1000000.0;  
  }
}

double aubio_utime(void)
{
  return (rdtsc() / aubio_cpu_frequency);
}

double rdtsc(void)
{
#if 0
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (double) t.QuadPart;
#else
  return 0.;
#endif
}

double aubio_estimate_cpu_frequency(void)
{
#if 0
  LONGLONG start, stop, ticks;
  unsigned int before, after, delta;
  double freq;

  start = rdtsc();
  stop = start;
  before = aubio_curtime();
  after = before;

  while (1) {
    if (after - before > 1000) {
    break;
    }
    after = aubio_curtime();
    stop = rdtsc();
  }

  delta = after - before;
  ticks = stop - start;

  freq = 1000 * ticks / delta;

  return freq;

#else
#if 0
  unsigned int before, after;
  LARGE_INTEGER start, stop;

  before = aubio_curtime();
  QueryPerformanceCounter(&start);

  Sleep(1000);
  
  after = aubio_curtime();
  QueryPerformanceCounter(&stop);

  return (double) 1000 * (stop.QuadPart - start.QuadPart) / (after - before);
#endif
  return 0;
#endif
}



#elif defined(MACOS9)
/*=============================================================*/
/*                                                             */
/*                           MacOS 9                           */
/*                                                             */
/*=============================================================*/


/***************************************************************
 *
 *               Timer
 */

struct _aubio_timer_t 
{
    TMTask myTmTask;
  long msec;
  unsigned int start;
  unsigned int count;
  int isInstalled;
  aubio_timer_callback_t callback;
  void* data;
  int auto_destroy;
};

static TimerUPP myTimerUPP;

void
_timerCallback(aubio_timer_t *timer)
{
    int cont;
  cont = (*timer->callback)(timer->data, aubio_curtime() - timer->start);
  if (cont) {
    PrimeTime((QElemPtr)timer, timer->msec);
    } else {
        timer->isInstalled = 0;
    }
  timer->count++;
}

aubio_timer_t* 
new_aubio_timer(int msec, aubio_timer_callback_t callback, void* data, 
           int new_thread, int auto_destroy)
{
  aubio_timer_t* timer = AUBIO_NEW(aubio_timer_t);
  if (timer == NULL) {
    AUBIO_ERR( "Out of memory");     
    return NULL;
  }

    if (!myTimerUPP)
        myTimerUPP = NewTimerProc(_timerCallback);

  /* setup tmtask */
    timer->myTmTask.tmAddr = myTimerUPP;
    timer->myTmTask.qLink = NULL;
    timer->myTmTask.qType = 0;
    timer->myTmTask.tmCount = 0L;
    timer->myTmTask.tmWakeUp = 0L;
    timer->myTmTask.tmReserved = 0L;

  timer->callback = callback;

  timer->msec = msec;
  timer->data = data;
  timer->start = aubio_curtime();
  timer->isInstalled = 1;
  timer->count = 0;
  timer->auto_destroy = auto_destroy;
  
  InsXTime((QElemPtr)timer);
  PrimeTime((QElemPtr)timer, msec);

  return timer;
}

int 
delete_aubio_timer(aubio_timer_t* timer)
{
    if (timer->isInstalled) {
        RmvTime((QElemPtr)timer);
    }
  AUBIO_FREE(timer);
  return AUBIO_OK;
}

int 
aubio_timer_join(aubio_timer_t* timer)
{
    if (timer->isInstalled) {
        int count = timer->count;
        /* wait until count has incremented */
        while (count == timer->count) {}
    }
  return AUBIO_OK;
}

/***************************************************************
 *
 *               Time
 */
#define kTwoPower32 (4294967296.0)      /* 2^32 */

void aubio_time_config(void)
{
}

unsigned int aubio_curtime()
{
    /* could be optimized by not going though a double */
    UnsignedWide    uS;
    double mSf;
    unsigned int ms;
    
    Microseconds(&uS);
    
  mSf = ((((double) uS.hi) * kTwoPower32) + uS.lo)/1000.0f;
  
  ms = mSf;
  
  return (ms);
}



#else

/*=============================================================*/
/*                                                             */
/*                           POSIX                             */
/*                                                             */
/*=============================================================*/

#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>


/***************************************************************
 *
 *               Timer
 */
 
struct _aubio_timer_t 
{
  long msec;
  aubio_timer_callback_t callback;
  void* data;
  pthread_t thread;
  int cont;
  int auto_destroy;
};

void* 
aubio_timer_start(void *data)
{
  int count = 0;
  int cont = 1;
  long start;
  long delay;
  aubio_timer_t* timer;
  timer = (aubio_timer_t*) data;

  /* keep track of the start time for absolute positioning */
  start = aubio_curtime();

  while (cont) {

    /* do whatever we have to do */
    cont = (*timer->callback)(timer->data, aubio_curtime() - start);

    count++;

    /* to avoid incremental time errors, calculate the delay between
       two callbacks bringing in the "absolute" time (count *
       timer->msec) */
    delay = (count * timer->msec) - (aubio_curtime() - start);
    if (delay > 0) {
      usleep(delay * 1000);
    }

    cont &= timer->cont;
  }

  AUBIO_DBG( "Timer thread finished");
  if (timer->thread != 0) {
    pthread_exit(NULL);
  }

  if (timer->auto_destroy) {
    AUBIO_FREE(timer);
  }

  return NULL;
}

aubio_timer_t* 
new_aubio_timer(int msec, aubio_timer_callback_t callback, void* data, 
           int new_thread, int auto_destroy)
{
  aubio_timer_t* timer = AUBIO_NEW(aubio_timer_t);
  if (timer == NULL) {
    AUBIO_ERR( "Out of memory");     
    return NULL;
  }
  timer->msec = msec;
  timer->callback = callback;
  timer->data = data;
  timer->cont = 1;
  timer->thread = 0;
  timer->auto_destroy = auto_destroy;

  if (new_thread) {
    if (pthread_create(&timer->thread, NULL, aubio_timer_start, (void*) timer)) {
      AUBIO_ERR( "Failed to create the timer thread");
      AUBIO_FREE(timer);
      return NULL;
    }
  } else {
    aubio_timer_start((void*) timer);
  }
  return timer;
}

int 
delete_aubio_timer(aubio_timer_t* timer)
{
  timer->cont = 0;
  aubio_timer_join(timer);
  AUBIO_DBG( "Deleted player thread\n");
  AUBIO_FREE(timer);
  return AUBIO_OK;
}

int 
aubio_timer_join(aubio_timer_t* timer)
{
  int err = 0;

  if (timer->thread != 0) {
    err = pthread_join(timer->thread, NULL);
  } else
    AUBIO_DBG( "Joined player thread\n");
  return (err == 0)? AUBIO_OK : AUBIO_FAIL;
}


/***************************************************************
 *
 *               Time
 */

static double aubio_cpu_frequency = -1.0;

double rdtsc(void);
double aubio_estimate_cpu_frequency(void);

void aubio_time_config(void)
{
  if (aubio_cpu_frequency < 0.0) {
    aubio_cpu_frequency = aubio_estimate_cpu_frequency() / 1000000.0;  
  }
}

unsigned int aubio_curtime()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return now.tv_sec * 1000 + now.tv_usec / 1000;
}

double aubio_utime(void)
{
  return (rdtsc() / aubio_cpu_frequency);
}

#if !defined(__i386__)

double rdtsc(void)
{
  return 0.0;
}

double aubio_estimate_cpu_frequency(void)
{
  return 1.0;
}

#else

double rdtsc(void)
{
  unsigned int a, b;

  __asm__ ("rdtsc" : "=a" (a), "=d" (b));
  return (double)b * (double)0x10000 * (double)0x10000 + a;
}

double aubio_estimate_cpu_frequency(void)
{
  double start, stop;
  unsigned int a0, b0, a1, b1;
  unsigned int before, after;

  before = aubio_curtime();  
  __asm__ ("rdtsc" : "=a" (a0), "=d" (b0));

  sleep(1);
  
  after = aubio_curtime();
  __asm__ ("rdtsc" : "=a" (a1), "=d" (b1));


  start = (double)b0 * (double)0x10000 * (double)0x10000 + a0;
  stop = (double)b1 * (double)0x10000 * (double)0x10000 + a1;

  return 1000 * (stop - start) / (after - before);
}

#endif

#endif
