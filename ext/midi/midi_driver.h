/* 
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

/* this file orginally taken from :
 * FluidSynth - A Software Synthesizer
 * Copyright (C) 2003  Peter Hanappe and others.
 */


/** \file
 * generic midi driver
 */

#ifndef _AUBIO_MDRIVER_H
#define _AUBIO_MDRIVER_H

typedef void aubio_settings_t;

typedef int (*handle_midi_event_func_t)(void* data, aubio_midi_event_t* event);

/** aubio_midi_driver_t */
typedef struct _aubio_midi_driver_t aubio_midi_driver_t;

struct _aubio_midi_driver_t 
{
  char* name;
  handle_midi_event_func_t handler;
  void* data;
};

//aubio_midi_driver_t* new_aubio_midi_driver(aubio_settings_t* settings, 
aubio_midi_driver_t* new_aubio_midi_driver(char * name, 
    handle_midi_event_func_t handler, 
    void* event_handler_data);
void del_aubio_midi_driver(aubio_midi_driver_t* driver);
void aubio_midi_driver_settings(aubio_settings_t* settings);

#include "config.h"
#if JACK_SUPPORT
void aubio_midi_direct_output(aubio_midi_driver_t * dev, aubio_midi_event_t * event); 
#endif

#endif  /* _AUBIO_AUDRIVER_H */
