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

/** \file
 * midi event structure
 */

#ifndef _AUBIO_MIDI_EVENT_H
#define _AUBIO_MIDI_EVENT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_midi_event_t aubio_midi_event_t;

/*
 * aubio_midi_event_t
 */
struct _aubio_midi_event_t {
  aubio_midi_event_t* next; /**< Don't use it, it will dissappear. Used in midi tracks.  */
  unsigned int dtime;       /**< Delay (ticks) between this and previous event. midi tracks. */
  unsigned char type;       /**< MIDI event type */
  unsigned char channel;    /**< MIDI channel */
  unsigned int param1;      /**< First parameter */
  unsigned int param2;      /**< Second parameter */
};

aubio_midi_event_t* new_aubio_midi_event(void);
int del_aubio_midi_event(aubio_midi_event_t* event);
int aubio_midi_event_set_type(aubio_midi_event_t* evt, int type);
int aubio_midi_event_get_type(aubio_midi_event_t* evt);
int aubio_midi_event_set_channel(aubio_midi_event_t* evt, int chan);
int aubio_midi_event_get_channel(aubio_midi_event_t* evt);
int aubio_midi_event_get_key(aubio_midi_event_t* evt);
int aubio_midi_event_set_key(aubio_midi_event_t* evt, int key);
int aubio_midi_event_get_velocity(aubio_midi_event_t* evt);
int aubio_midi_event_set_velocity(aubio_midi_event_t* evt, int vel);
int aubio_midi_event_get_control(aubio_midi_event_t* evt);
int aubio_midi_event_set_control(aubio_midi_event_t* evt, int ctrl);
int aubio_midi_event_get_value(aubio_midi_event_t* evt);
int aubio_midi_event_set_value(aubio_midi_event_t* evt, int val);
int aubio_midi_event_get_program(aubio_midi_event_t* evt);
int aubio_midi_event_set_program(aubio_midi_event_t* evt, int val);
int aubio_midi_event_get_pitch(aubio_midi_event_t* evt);
int aubio_midi_event_set_pitch(aubio_midi_event_t* evt, int val);
int aubio_midi_event_length(unsigned char status);

#ifdef __cplusplus
}
#endif

#endif/*_AUBIO_MIDI_EVENT_H*/
