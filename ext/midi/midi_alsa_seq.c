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
 * Midi driver for the Advanced Linux Sound Architecture (sequencer mode)
 */


#include "aubio_priv.h"
#include "midi.h"
#include "midi_event.h"
#include "midi_parser.h"
#include "midi_driver.h"
#include "config.h"

#if ALSA_SUPPORT

#define ALSA_PCM_NEW_HW_PARAMS_API
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/poll.h>
/* #include <errno.h> //perror is in stdio.h */

#define AUBIO_ALSA_DEFAULT_SEQ_DEVICE   "default"

#define AUBIO_ALSA_BUFFER_LENGTH 512

/* SCHED_FIFO priorities for ALSA threads (see pthread_attr_setschedparam) */
#define ALSA_RAWMIDI_SCHED_PRIORITY 90
#define ALSA_SEQ_SCHED_PRIORITY 90


/** aubio_alsa_seq_driver_t */
typedef struct {
    aubio_midi_driver_t driver;
    snd_seq_t *seq_handle;
    int seq_port;
    struct pollfd *pfd;
    int npfd;
    pthread_t thread;
    int status;
} aubio_alsa_seq_driver_t;

aubio_midi_driver_t* new_aubio_alsa_seq_driver(//aubio_settings_t* settings, 
        handle_midi_event_func_t handler, 
        void* data);
int del_aubio_alsa_seq_driver(aubio_midi_driver_t* p);
static void* aubio_alsa_seq_run(void* d);

//void aubio_alsa_seq_driver_settings(aubio_settings_t* settings)
//{
//  aubio_settings_register_str(settings, "midi.alsa_seq.device", "default", 0, NULL, NULL);
//  aubio_settings_register_str(settings, "midi.alsa_seq.id", "pid", 0, NULL, NULL);
//}

/** new_aubio_alsa_seq_driver */
aubio_midi_driver_t*  new_aubio_alsa_seq_driver(//aubio_settings_t* settings, 
        handle_midi_event_func_t handler, void* data)
{
    int i, err;                     
    aubio_alsa_seq_driver_t* dev;   /**< object to return */
    pthread_attr_t attr;            /**< sequencer thread */
    int sched = SCHED_FIFO;         /**< default scheduling policy */
    struct sched_param priority;    /**< scheduling priority settings */
    int count;                      /**< number of MIDI file descriptors */
    struct pollfd *pfd = NULL;      /**< poll file descriptor array (copied in dev->pfd) */
    char* device = NULL;            /**< the device name */
    char* id = NULL;
    char full_id[64];
    char full_name[64];

    /* not much use doing anything */
    if (handler == NULL) {
        AUBIO_ERR( "Invalid argument");
        return NULL;
    }

    /* allocate the device */
    dev = AUBIO_NEW(aubio_alsa_seq_driver_t);
    if (dev == NULL) {
        AUBIO_ERR( "Out of memory");
        return NULL;
    }
    AUBIO_MEMSET(dev, 0, sizeof(aubio_alsa_seq_driver_t));
    dev->seq_port = -1;
    dev->driver.data = data;
    dev->driver.handler = handler;

    /* get the device name. if none is specified, use the default device. */
    //aubio_settings_getstr(settings, "midi.alsa_seq.device", &device);
    if (device == NULL) {
        device = "default";
    }

    /* open the sequencer INPUT only, non-blocking */
    //if ((err = snd_seq_open(&dev->seq_handle, device, SND_SEQ_OPEN_INPUT,
    if ((err = snd_seq_open(&dev->seq_handle, device, SND_SEQ_OPEN_DUPLEX,
                    SND_SEQ_NONBLOCK)) < 0) {
        AUBIO_ERR( "Error opening ALSA sequencer");
        goto error_recovery;
    }

    /* get # of MIDI file descriptors */
    count = snd_seq_poll_descriptors_count(dev->seq_handle, POLLIN);
    if (count > 0) {        /* make sure there are some */
        pfd = AUBIO_MALLOC(sizeof (struct pollfd) * count);
        dev->pfd = AUBIO_MALLOC(sizeof (struct pollfd) * count);
        /* grab file descriptor POLL info structures */
        count = snd_seq_poll_descriptors(dev->seq_handle, pfd, count, POLLIN);
    }

    for (i = 0; i < count; i++) {        /* loop over file descriptors */
        /* copy the input FDs */
        if (pfd[i].events & POLLIN) { /* use only the input FDs */
            dev->pfd[dev->npfd].fd = pfd[i].fd;
            dev->pfd[dev->npfd].events = POLLIN; 
            dev->pfd[dev->npfd].revents = 0; 
            dev->npfd++;
        }
    }
    AUBIO_FREE(pfd);

    //aubio_settings_getstr(settings, "midi.alsa_seq.id", &id);

    if (id != NULL) {
        if (AUBIO_STRCMP(id, "pid") == 0) {
            snprintf(full_id, 64, "aubio (%d)", getpid());
            snprintf(full_name, 64, "aubio_port (%d)", getpid());
        } else {
            snprintf(full_id, 64, "aubio (%s)", id);
            snprintf(full_name, 64, "aubio_port (%s)", id);
        }
    } else {
        snprintf(full_id, 64, "aubio");
        snprintf(full_name, 64, "aubio_port");
    }

    /* set the client name */
    snd_seq_set_client_name (dev->seq_handle, full_id);

    if ((dev->seq_port = snd_seq_create_simple_port (dev->seq_handle,
                    full_name,
                    SND_SEQ_PORT_CAP_WRITE | SND_SEQ_PORT_CAP_SUBS_WRITE |
                    SND_SEQ_PORT_CAP_READ | SND_SEQ_PORT_CAP_SUBS_READ | 
                    SND_SEQ_PORT_CAP_DUPLEX,
                    SND_SEQ_PORT_TYPE_APPLICATION)) < 0)
    {
        AUBIO_ERR( "Error creating ALSA sequencer port");
        goto error_recovery;
    }

    dev->status = AUBIO_MIDI_READY;

    /* create the midi thread */
    if (pthread_attr_init(&attr)) {
        AUBIO_ERR( "Couldn't initialize midi thread attributes");
        goto error_recovery;
    }

    /* use fifo scheduling. if it fails, use default scheduling. */
    while (1) {
        err = pthread_attr_setschedpolicy(&attr, sched);
        if (err) {
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
        priority.sched_priority = (sched == SCHED_FIFO) ? ALSA_SEQ_SCHED_PRIORITY : 0;
        pthread_attr_setschedparam (&attr, &priority);

        err = pthread_create(&dev->thread, &attr, aubio_alsa_seq_run, (void*) dev);
        if (err) {
            AUBIO_ERR( "Couldn't set high priority scheduling for the MIDI input");
            if (sched == SCHED_FIFO) {
                sched = SCHED_OTHER;
                continue;
            } else {
                //AUBIO_LOG(AUBIO_PANIC, "Couldn't create the midi thread.");
                AUBIO_ERR( "Couldn't create the midi thread.");
                goto error_recovery;
            }
        }
        break;
    }
    return (aubio_midi_driver_t*) dev;


error_recovery:
    del_aubio_alsa_seq_driver((aubio_midi_driver_t*) dev);
    return NULL;
}

/** del_aubio_alsa_seq_driver */
int del_aubio_alsa_seq_driver(aubio_midi_driver_t* p)
{
    aubio_alsa_seq_driver_t* dev;

    dev = (aubio_alsa_seq_driver_t*) p;
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
    if (dev->seq_port >= 0) {
        snd_seq_delete_simple_port (dev->seq_handle, dev->seq_port);
    }
    if (dev->seq_handle) {
        snd_seq_drain_output(dev->seq_handle);
        snd_seq_close(dev->seq_handle);
    }
    AUBIO_FREE(dev);
    return AUBIO_OK;
}

/** aubio_alsa_seq_run */
void* aubio_alsa_seq_run(void* d)
{
    int n;//, i;
    snd_seq_event_t *seq_ev;
    aubio_midi_event_t evt;
    aubio_alsa_seq_driver_t* dev = (aubio_alsa_seq_driver_t*) d;

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
        n = poll(dev->pfd, dev->npfd, 1); /* use a 1 milliseconds timeout */
        if (n < 0) {
            perror("poll");
        } else if (n > 0) {

            /* read new events from the midi input port */
            while ((n = snd_seq_event_input(dev->seq_handle, &seq_ev)) >= 0)
            {
                switch (seq_ev->type)
                {
                    case SND_SEQ_EVENT_NOTEON:
                        evt.type = NOTE_ON;
                        evt.channel = seq_ev->data.note.channel;
                        evt.param1 = seq_ev->data.note.note;
                        evt.param2 = seq_ev->data.note.velocity;
                        break;
                    case SND_SEQ_EVENT_NOTEOFF:
                        evt.type = NOTE_OFF;
                        evt.channel = seq_ev->data.note.channel;
                        evt.param1 = seq_ev->data.note.note;
                        evt.param2 = seq_ev->data.note.velocity;
                        break;
                    case SND_SEQ_EVENT_KEYPRESS:
                        evt.type = KEY_PRESSURE;
                        evt.channel = seq_ev->data.note.channel;
                        evt.param1 = seq_ev->data.note.note;
                        evt.param2 = seq_ev->data.note.velocity;
                        break;
                    case SND_SEQ_EVENT_CONTROLLER:
                        evt.type = CONTROL_CHANGE;
                        evt.channel = seq_ev->data.control.channel;
                        evt.param1 = seq_ev->data.control.param;
                        evt.param2 = seq_ev->data.control.value;
                        break;
                    case SND_SEQ_EVENT_PITCHBEND:
                        evt.type = PITCH_BEND;
                        evt.channel = seq_ev->data.control.channel;
                        /* ALSA pitch bend is -8192 - 8191, we adjust it here */
                        evt.param1 = seq_ev->data.control.value + 8192;
                        break;
                    case SND_SEQ_EVENT_PGMCHANGE:
                        evt.type = PROGRAM_CHANGE;
                        evt.channel = seq_ev->data.control.channel;
                        evt.param1 = seq_ev->data.control.value;
                        break;
                    case SND_SEQ_EVENT_CHANPRESS:
                        evt.type = CHANNEL_PRESSURE;
                        evt.channel = seq_ev->data.control.channel;
                        evt.param1 = seq_ev->data.control.value;
                        break;
                    default:
                        continue;        /* unhandled event, next loop iteration */
                }

                /* send the events to the next link in the chain */
                (*dev->driver.handler)(dev->driver.data, &evt);
                
                /* dump input on output */
                //snd_seq_ev_set_source(new_ev, dev->seq_port);
                //snd_seq_ev_set_dest(seq_ev,dev->seq_handle,dev->seq_client);
                //snd_seq_ev_set_subs(new_ev);
                //snd_seq_ev_set_direct(new_ev);
                //snd_seq_event_output(dev->seq_handle, new_ev);
                //snd_seq_drain_output(dev->seq_handle);

            }
        }

        if ((n < 0) && (n != -EAGAIN)) {
            AUBIO_ERR( "Error occured while reading ALSA sequencer events");
            dev->status = AUBIO_MIDI_DONE;
        }

//        /* added by piem to handle new data to output */
//        while (/* get new data, but from where ??? (n = snd_seq_event_output(dev->seq_handle, seq_ev)) >= 0*/ )
//        {
//            /* dump input on output */
//            snd_seq_ev_set_source(new_ev, dev->seq_port);
//            //snd_seq_ev_set_dest(seq_ev,dev->seq_handle,dev->seq_client);
//            snd_seq_ev_set_subs(new_ev);
//            snd_seq_ev_set_direct(new_ev);
//            snd_seq_event_output(dev->seq_handle, new_ev);
//            snd_seq_drain_output(dev->seq_handle);
//        }

    }
    pthread_exit(NULL);
}


snd_seq_event_t ev;

void aubio_midi_direct_output(aubio_midi_driver_t * d, aubio_midi_event_t * event) 
{
    aubio_alsa_seq_driver_t* dev = (aubio_alsa_seq_driver_t*) d;
    switch(event->type) 
    {
        case NOTE_ON:
            ev.type = SND_SEQ_EVENT_NOTEON;
            ev.data.note.channel  = event->channel;
            ev.data.note.note     = event->param1;
            ev.data.note.velocity = event->param2;
            //AUBIO_ERR( "NOTE_ON %d\n", event->param1);
            break;
        case NOTE_OFF:
            ev.type = SND_SEQ_EVENT_NOTEOFF;
            ev.data.note.channel  = event->channel;
            ev.data.note.note     = event->param1;
            ev.data.note.velocity = event->param2;
            //AUBIO_ERR( "NOTE_OFF %d\n", event->param1);
            break;
        default:
            break;
    }
    if (ev.type == SND_SEQ_EVENT_NOTEOFF || ev.type == SND_SEQ_EVENT_NOTEON ) { 
        snd_seq_ev_set_subs(&ev);
        snd_seq_ev_set_direct(&ev);
        snd_seq_ev_set_source(&ev, dev->seq_port);
        snd_seq_event_output_direct(dev->seq_handle, &ev);
    }
}

#endif /* #if ALSA_SUPPORT */
