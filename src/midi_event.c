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
#include "midi_event.h"
#include "midi.h"

/******************************************************
 *
 *     aubio_event_t
 */

/*
 * new_aubio_midi_event
 */
aubio_midi_event_t* new_aubio_midi_event()
{
  aubio_midi_event_t* evt;
  evt = AUBIO_NEW(aubio_midi_event_t);
  if (evt == NULL) {
    AUBIO_ERR( "Out of memory");
    return NULL;
  }
  evt->dtime = 0;
  evt->type = 0;
  evt->channel = 0;
  evt->param1 = 0;
  evt->param2 = 0;
  evt->next = NULL;
  return evt;
}

/** del_aubio_midi_event */
int del_aubio_midi_event(aubio_midi_event_t* evt)
{
  aubio_midi_event_t *temp;
  while(evt)
  {
    temp = evt->next;
    AUBIO_FREE(evt);
    evt = temp;
  }
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_type
 */
int aubio_midi_event_get_type(aubio_midi_event_t* evt)
{
  return evt->type;
}

/*
 * aubio_midi_event_set_type
 */
int aubio_midi_event_set_type(aubio_midi_event_t* evt, int type)
{
  evt->type = type;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_channel
 */
int aubio_midi_event_get_channel(aubio_midi_event_t* evt)
{
  return evt->channel;
}

/*
 * aubio_midi_event_set_channel
 */
int aubio_midi_event_set_channel(aubio_midi_event_t* evt, int chan)
{
  evt->channel = chan;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_key
 */
int aubio_midi_event_get_key(aubio_midi_event_t* evt)
{
  return evt->param1;
}

/*
 * aubio_midi_event_set_key
 */
int aubio_midi_event_set_key(aubio_midi_event_t* evt, int v)
{
  evt->param1 = v;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_velocity
 */
int aubio_midi_event_get_velocity(aubio_midi_event_t* evt)
{
  return evt->param2;
}

/*
 * aubio_midi_event_set_velocity
 */
int aubio_midi_event_set_velocity(aubio_midi_event_t* evt, int v)
{
  evt->param2 = v;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_control
 */
int aubio_midi_event_get_control(aubio_midi_event_t* evt)
{
  return evt->param1;
}

/*
 * aubio_midi_event_set_control
 */
int aubio_midi_event_set_control(aubio_midi_event_t* evt, int v)
{
  evt->param1 = v;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_value
 */
int aubio_midi_event_get_value(aubio_midi_event_t* evt)
{
  return evt->param2;
}

/*
 * aubio_midi_event_set_value
 */
int aubio_midi_event_set_value(aubio_midi_event_t* evt, int v)
{
  evt->param2 = v;
  return AUBIO_OK;
}

int aubio_midi_event_get_program(aubio_midi_event_t* evt)
{
  return evt->param1;
}

int aubio_midi_event_set_program(aubio_midi_event_t* evt, int val)
{
  evt->param1 = val;
  return AUBIO_OK;
}

int aubio_midi_event_get_pitch(aubio_midi_event_t* evt)
{
  return evt->param1;
}

int aubio_midi_event_set_pitch(aubio_midi_event_t* evt, int val)
{
  evt->param1 = val;
  return AUBIO_OK;
}

/*
 * aubio_midi_event_get_param1
 */
/* int aubio_midi_event_get_param1(aubio_midi_event_t* evt) */
/* { */
/*   return evt->param1; */
/* } */

/*
 * aubio_midi_event_set_param1
 */
/* int aubio_midi_event_set_param1(aubio_midi_event_t* evt, int v) */
/* { */
/*   evt->param1 = v; */
/*   return AUBIO_OK; */
/* } */

/*
 * aubio_midi_event_get_param2
 */
/* int aubio_midi_event_get_param2(aubio_midi_event_t* evt) */
/* { */
/*   return evt->param2; */
/* } */

/*
 * aubio_midi_event_set_param2
 */
/* int aubio_midi_event_set_param2(aubio_midi_event_t* evt, int v) */
/* { */
/*   evt->param2 = v; */
/*   return AUBIO_OK; */
/* } */

