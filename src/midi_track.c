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
#include "midi.h"
#include "midi_event.h"
#include "midi_track.h"
#include "midi_player.h"


/** new_aubio_track */
aubio_track_t* new_aubio_track(int num)
{
  aubio_track_t* track;
  track = AUBIO_NEW(aubio_track_t);
  if (track == NULL) {
    return NULL;
  }
  track->name = NULL;
  track->num = num;
  track->first = NULL;
  track->cur = NULL;
  track->last = NULL;
  track->ticks = 0;
  return track;
}

/** del_aubio_track */
int del_aubio_track(aubio_track_t* track)
{
  if (track->name != NULL) {
    AUBIO_FREE(track->name);
  }
  if (track->first != NULL) {
    del_aubio_midi_event(track->first);
  }
  AUBIO_FREE(track);
  return AUBIO_OK;
}

/** aubio_track_set_name */
int aubio_track_set_name(aubio_track_t* track, char* name)
{
  int len;
  if (track->name != NULL) {
    AUBIO_FREE(track->name);
  }
  if (name == NULL) {
    track->name = NULL;
    return AUBIO_OK;  
  }
  len = AUBIO_STRLEN(name);
  track->name = AUBIO_MALLOC(len + 1);
  if (track->name == NULL) {
    AUBIO_ERR( "Out of memory");
    return AUBIO_FAIL;
  }
  AUBIO_STRCPY(track->name, name);
  return AUBIO_OK;  
}

/** aubio_track_get_name */
char* aubio_track_get_name(aubio_track_t* track)
{
  return track->name;
}

/** aubio_track_get_duration */
int aubio_track_get_duration(aubio_track_t* track)
 {
  int time = 0;
  aubio_midi_event_t* evt = track->first;
  while (evt != NULL) {
    time += evt->dtime;
    evt = evt->next;
  }
  return time;
}

/** aubio_track_count_events  */
int aubio_track_count_events(aubio_track_t* track, int* on, int* off)
{
  aubio_midi_event_t* evt = track->first;
  while (evt != NULL) {
    if (evt->type == NOTE_ON) {
      (*on)++;
    } else if (evt->type == NOTE_OFF) {
      (*off)++;
    }
    evt = evt->next;
  }
  return AUBIO_OK;
}

/*
 * aubio_track_add_event
 */
int aubio_track_add_event(aubio_track_t* track, aubio_midi_event_t* evt)
{
  evt->next = NULL;
  if (track->first == NULL) {
    track->first = evt;
    track->cur = evt;
    track->last = evt;
  } else {
    track->last->next = evt;
    track->last = evt;
  }
  return AUBIO_OK;
}

/*
 * aubio_track_first_event
 */
aubio_midi_event_t* aubio_track_first_event(aubio_track_t* track)
{
  track->cur = track->first;
  return track->cur;
}

/*
 * aubio_track_next_event
 */
aubio_midi_event_t* aubio_track_next_event(aubio_track_t* track)
{
  if (track->cur != NULL) {
    track->cur = track->cur->next;
  }
  return track->cur;
}

/*
 * aubio_track_reset
 */
  int 
aubio_track_reset(aubio_track_t* track)
{
  track->ticks = 0;
  track->cur = track->first;
  return AUBIO_OK;
}

