/* 
 * Copyright 2004 Paul Brossier
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

/* this file originally taken from Fluidsynth, Peter Hanappe and others. */
 
/** \file
 * Midi driver for the Advanced Linux Sound Architecture
 */


#include "aubio_priv.h"
#include "midi.h"
#include "midi_event.h"
#include "midi_parser.h"
#include "midi_driver.h"

#if ALSA_SUPPORT

#define ALSA_PCM_NEW_HW_PARAMS_API
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/poll.h>
/* #include <errno.h> //perror is in stdio.h */

#include "config.h"

#define AUBIO_ALSA_DEFAULT_MIDI_DEVICE  "default"

/** \bug double define? */
#define AUBIO_ALSA_BUFFER_LENGTH 512

/* SCHED_FIFO priorities for ALSA threads (see pthread_attr_setschedparam) */
#define ALSA_RAWMIDI_SCHED_PRIORITY 90
#define ALSA_SEQ_SCHED_PRIORITY 90

/** aubio_midi_alsa_raw_driver_t */
typedef struct {
  aubio_midi_driver_t driver;
  snd_rawmidi_t *rawmidi_in;
  snd_rawmidi_t *rawmidi_out;
  struct pollfd *pfd;
  int npfd;
  pthread_t thread;
  int status;
  unsigned char buffer[AUBIO_ALSA_BUFFER_LENGTH];
  aubio_midi_parser_t* parser;
} aubio_midi_alsa_raw_driver_t;

aubio_midi_driver_t* new_aubio_midi_alsa_raw_driver(//aubio_settings_t* settings, 
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_midi_alsa_raw_driver(aubio_midi_driver_t* p);
static void* aubio_midi_alsa_raw_run(void* d);


/**************************************************************
 *
 *        Alsa MIDI driver
 *
 */

//void aubio_midi_alsa_raw_driver_settings(aubio_settings_t* settings)
//{
//  aubio_settings_register_str(settings, "midi.alsa.device", "default", 0, NULL, NULL);
//}

/** new_aubio_midi_alsa_raw_driver */
aubio_midi_driver_t* new_aubio_midi_alsa_raw_driver(
    //aubio_settings_t* settings, 
    handle_midi_event_func_t handler, 
    void* data)
{
  int i, err;
  aubio_midi_alsa_raw_driver_t* dev;
  pthread_attr_t attr;
  int sched = SCHED_FIFO;
  struct sched_param priority;
  int count;
  struct pollfd *pfd = NULL;
  char* device = NULL;

  /* not much use doing anything */
  if (handler == NULL) {
    AUBIO_ERR("Invalid argument");
    return NULL;
  }

  /* allocate the device */
  dev = AUBIO_NEW(aubio_midi_alsa_raw_driver_t);
  if (dev == NULL) {
    AUBIO_ERR( "Out of memory");
    return NULL;
  }
  AUBIO_MEMSET(dev, 0, sizeof(aubio_midi_alsa_raw_driver_t));

  dev->driver.handler = handler;
  dev->driver.data = data;

  /* allocate one event to store the input data */
  dev->parser = new_aubio_midi_parser();
  if (dev->parser == NULL) {
    AUBIO_ERR( "Out of memory");
    goto error_recovery;
  }

  /* get the device name. if none is specified, use the default device. */
  //aubio_settings_getstr(settings, "midi.alsa.device", &device);
  if (device == NULL) {
    device = "default";
  }

  /* open the hardware device. only use midi in. */
  if ((err = snd_rawmidi_open(&dev->rawmidi_in, NULL, device, SND_RAWMIDI_NONBLOCK)) < 0) {
  //if ((err = snd_rawmidi_open(&dev->rawmidi_in, &dev->rawmidi_out, device, SND_RAWMIDI_NONBLOCK)) < 0) {
    AUBIO_ERR( "Error opening ALSA raw MIDI IN port");
    goto error_recovery;
  }

  /* get # of MIDI file descriptors */
  count = snd_rawmidi_poll_descriptors_count(dev->rawmidi_in);
  if (count > 0) {		/* make sure there are some */
    pfd = AUBIO_MALLOC(sizeof (struct pollfd) * count);
    dev->pfd = AUBIO_MALLOC(sizeof (struct pollfd) * count);
    /* grab file descriptor POLL info structures */
    count = snd_rawmidi_poll_descriptors(dev->rawmidi_in, pfd, count);
  }

  /* copy the input FDs */
  for (i = 0; i < count; i++) {		/* loop over file descriptors */
    if (pfd[i].events & POLLIN) { /* use only the input FDs */
      dev->pfd[dev->npfd].fd = pfd[i].fd;
      dev->pfd[dev->npfd].events = POLLIN; 
      dev->pfd[dev->npfd].revents = 0; 
      dev->npfd++;
    }
  }
  AUBIO_FREE(pfd);


  
  dev->status = AUBIO_MIDI_READY;

  /* create the midi thread */
  if (pthread_attr_init(&attr)) {
    AUBIO_ERR( "Couldn't initialize midi thread attributes");
    goto error_recovery;
  }

  /* Was: "use fifo scheduling. if it fails, use default scheduling." */
  /* Now normal scheduling is used by default for the MIDI thread. The reason is,
   * that fluidsynth works better with low latencies under heavy load, if only the 
   * audio thread is prioritized.
   * With MIDI at ordinary priority, that could result in individual notes being played
   * a bit late. On the other hand, if the audio thread is delayed, an audible dropout
   * is the result.
   * To reproduce this: Edirol UA-1 USB-MIDI interface, four buffers
   * with 45 samples each (roughly 4 ms latency), ravewave soundfont. -MN
   */ 

  /* Not so sure anymore. We're losing MIDI data, if we can't keep up with
   * the speed it is generated. */
  /* AUBIO_MSG("Note: High-priority scheduling for the MIDI thread was intentionally disabled.");
      sched=SCHED_OTHER;*/

  while (1) {
    err = pthread_attr_setschedpolicy(&attr, sched);
    if (err) {
      //AUBIO_LOG(AUBIO_WARN, "Couldn't set high priority scheduling for the MIDI input");
      AUBIO_MSG( "Couldn't set high priority scheduling for the MIDI input");
      if (sched == SCHED_FIFO) {
        sched = SCHED_OTHER;
        continue;
      } else {
        AUBIO_ERR( "Couldn't set scheduling policy.");
        goto error_recovery;
      }
    }

    /* SCHED_FIFO will not be active without setting the priority */
    priority.sched_priority = (sched == SCHED_FIFO) ? 
      ALSA_RAWMIDI_SCHED_PRIORITY : 0;
    pthread_attr_setschedparam (&attr, &priority);
    err = pthread_create(&dev->thread, &attr, aubio_midi_alsa_raw_run, (void*) dev);
    if (err) {
      AUBIO_MSG( "Couldn't set high priority scheduling for the MIDI input");
      if (sched == SCHED_FIFO) {
        sched = SCHED_OTHER;
        continue;
      } else {
        AUBIO_ERR( "Couldn't create the midi thread.");
        goto error_recovery;
      }
    }
    break;
  }  
  return (aubio_midi_driver_t*) dev;

error_recovery:
  del_aubio_midi_alsa_raw_driver((aubio_midi_driver_t*) dev);
  return NULL;

}

/** del_aubio_midi_alsa_raw_driver */
int del_aubio_midi_alsa_raw_driver(aubio_midi_driver_t* p)
{
  aubio_midi_alsa_raw_driver_t* dev;

  dev = (aubio_midi_alsa_raw_driver_t*) p;
  if (dev == NULL) {
    return AUBIO_OK;
  }

  dev->status = AUBIO_MIDI_DONE;

  /* cancel the thread and wait for it before cleaning up */
  if (dev->thread) {
    if (pthread_cancel(dev->thread)) {
      AUBIO_ERR( "Failed to cancel the midi thread");
      return AUBIO_FAIL;
    }
    if (pthread_join(dev->thread, NULL)) {
      AUBIO_ERR( "Failed to join the midi thread");
      return AUBIO_FAIL;
    }
  }
  if (dev->rawmidi_in) {
    snd_rawmidi_drain(dev->rawmidi_in);
    snd_rawmidi_close(dev->rawmidi_in);
  }
  if (dev->rawmidi_out) {
    snd_rawmidi_drain(dev->rawmidi_out);
    snd_rawmidi_close(dev->rawmidi_in);
  }
  if (dev->parser != NULL) {
    del_aubio_midi_parser(dev->parser);
  }
  AUBIO_FREE(dev);
  return AUBIO_OK;
}

/** aubio_midi_alsa_raw_run */
void * aubio_midi_alsa_raw_run(void* d)
{
  int n, i;
  aubio_midi_event_t* evt;
  aubio_midi_alsa_raw_driver_t* dev = (aubio_midi_alsa_raw_driver_t*) d;

  /* make sure the other threads can cancel this thread any time */
  if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL)) {
    AUBIO_ERR( "Failed to set the cancel state of the midi thread");
    pthread_exit(NULL);
  }
  if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL)) {
    AUBIO_ERR( "Failed to set the cancel state of the midi thread");
    pthread_exit(NULL);
  }

  /* go into a loop until someone tells us to stop */
  dev->status = AUBIO_MIDI_LISTENING;
  while (dev->status == AUBIO_MIDI_LISTENING) {

    /* is there something to read? */
    /* use a 100 milliseconds timeout */
    n = poll(dev->pfd, dev->npfd, 100); 
    if (n < 0) {
      perror("poll");
    } else if (n > 0) {

      /* read new data */
      n = snd_rawmidi_read(dev->rawmidi_in, dev->buffer, 
          AUBIO_ALSA_BUFFER_LENGTH);
      if ((n < 0) && (n != -EAGAIN)) {
        AUBIO_ERR( "Failed to read the midi input");
        dev->status = AUBIO_MIDI_DONE;
      }

      /* let the parser convert the data into events */
      for (i = 0; i < n; i++) {
        evt = aubio_midi_parser_parse(dev->parser, dev->buffer[i]);
        if (evt != NULL) {
          (*dev->driver.handler)(dev->driver.data, evt);
        }
      }
    };
  }
  pthread_exit(NULL);
}

#endif /* #if ALSA_SUPPORT */
