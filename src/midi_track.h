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

#ifndef _AUBIO_MIDI_TRACK_H
#define _AUBIO_MIDI_TRACK_H

/** \file
 * midi track structure
 * 
 * \bug need public declaration ?
 */

#ifdef __cplusplus
extern "C" {
#endif

/** aubio_track_t */
struct _aubio_track_t {
  char* name;
  int num;
  aubio_midi_event_t *first;
  aubio_midi_event_t *cur;
  aubio_midi_event_t *last;
  unsigned int ticks;
};

typedef struct _aubio_track_t aubio_track_t;

aubio_track_t* new_aubio_track(int num);
int del_aubio_track(aubio_track_t* track);
int aubio_track_set_name(aubio_track_t* track, char* name);
char* aubio_track_get_name(aubio_track_t* track);
int aubio_track_add_event(aubio_track_t* track, aubio_midi_event_t* evt);
aubio_midi_event_t* aubio_track_first_event(aubio_track_t* track);
aubio_midi_event_t* aubio_track_next_event(aubio_track_t* track);
int aubio_track_get_duration(aubio_track_t* track);
int aubio_track_reset(aubio_track_t* track);
int aubio_track_count_events(aubio_track_t* track, int* on, int* off);


#define aubio_track_eot(track)  ((track)->cur == NULL)


#ifdef __cplusplus
}
#endif

#endif /*_AUBIO_MIDI_TRACK_H*/
