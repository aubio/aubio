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
/* This file originally tajke from 
 * FluidSynth - A Software Synthesizer
 *
 * Copyright (C) 2003  Peter Hanappe and others.
 */

#include "aubio_priv.h"
#include "midi_event.h"
#include "midi_driver.h"
//#include "settings.h"

/*
 * aubio_mdriver_definition
 */
struct aubio_mdriver_definition_t {
  char* name;
  aubio_midi_driver_t* (*new)(
      //aubio_settings_t* settings, 
      handle_midi_event_func_t event_handler, 
      void* event_handler_data);
  int (*free)(aubio_midi_driver_t* p);
  void (*settings)(aubio_settings_t* settings);
};


/* ALSA */
#if ALSA_SUPPORT
aubio_midi_driver_t* new_aubio_midi_alsa_raw_driver(
    //aubio_settings_t* settings,
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_midi_alsa_raw_driver(aubio_midi_driver_t* p);
void aubio_midi_alsa_raw_driver_settings(aubio_settings_t* settings);

aubio_midi_driver_t* new_aubio_alsa_seq_driver(
    //aubio_settings_t* settings,
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_alsa_seq_driver(aubio_midi_driver_t* p);
void aubio_alsa_seq_driver_settings(aubio_settings_t* settings);
#endif

/* OSS */
#if OSS_SUPPORT
aubio_midi_driver_t* new_aubio_oss_midi_driver(aubio_settings_t* settings,
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_oss_midi_driver(aubio_midi_driver_t* p);
//void aubio_oss_midi_driver_settings(aubio_settings_t* settings);
#endif

/* Windows MIDI service */
#if WINMIDI_SUPPORT
aubio_midi_driver_t* new_aubio_winmidi_driver(aubio_settings_t* settings,
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_winmidi_driver(aubio_midi_driver_t* p);
#endif

/* definitions for the MidiShare driver */
#if MIDISHARE_SUPPORT
aubio_midi_driver_t* new_aubio_midishare_midi_driver(aubio_settings_t* settings,
    handle_midi_event_func_t handler, 
    void* event_handler_data);
int del_aubio_midishare_midi_driver(aubio_midi_driver_t* p);
#endif


struct aubio_mdriver_definition_t aubio_midi_drivers[] = {
#if OSS_SUPPORT
  { "oss", 
    new_aubio_oss_midi_driver, 
    del_aubio_oss_midi_driver, 
    aubio_oss_midi_driver_settings },
#endif
#if ALSA_SUPPORT
  { "alsa_raw", 
    new_aubio_midi_alsa_raw_driver, 
    del_aubio_midi_alsa_raw_driver, 
    NULL /*aubio_midi_alsa_raw_driver_settings*/ },
  { "alsa_seq", 
    new_aubio_alsa_seq_driver, 
    del_aubio_alsa_seq_driver, 
    NULL /*aubio_alsa_seq_driver_settings*/ },
#endif
#if WINMIDI_SUPPORT
  { "winmidi", 
    new_aubio_winmidi_driver, 
    del_aubio_winmidi_driver, 
    NULL },
#endif
#if MIDISHARE_SUPPORT
  { "midishare", 
    new_aubio_midishare_midi_driver, 
    del_aubio_midishare_midi_driver, 
    NULL },
#endif
  { NULL, NULL, NULL, NULL }
};


void aubio_midi_driver_settings(aubio_settings_t* settings)
{  
  int i;

#if 0
  /* Set the default driver */
#if ALSA_SUPPORT
  aubio_settings_register_str(settings, "midi.driver", "alsa_seq", 0, NULL, NULL);
#elif OSS_SUPPORT
  aubio_settings_register_str(settings, "midi.driver", "oss", 0, NULL, NULL);
#elif WINMIDI_SUPPORT
  aubio_settings_register_str(settings, "midi.driver", "winmidi", 0, NULL, NULL);
#elif MIDISHARE_SUPPORT
  aubio_settings_register_str(settings, "midi.driver", "midishare", 0, NULL, NULL);
#else
  aubio_settings_register_str(settings, "midi.driver", "", 0, NULL, NULL);
#endif

  /* Add all drivers to the list of options */
#if ALSA_SUPPORT
  aubio_settings_add_option(settings, "midi.driver", "alsa_seq");
  aubio_settings_add_option(settings, "midi.driver", "alsa_raw");
#endif
#if OSS_SUPPORT
  aubio_settings_add_option(settings, "midi.driver", "oss");
#endif
#if WINMIDI_SUPPORT
  aubio_settings_add_option(settings, "midi.driver", "winmidi");
#endif
#if MIDISHARE_SUPPORT
  aubio_settings_add_option(settings, "midi.driver", "midishare");
#endif

#endif

  for (i = 0; aubio_midi_drivers[i].name != NULL; i++) {
    if (aubio_midi_drivers[i].settings != NULL) {
      aubio_midi_drivers[i].settings(settings);
    }
  }  
}

//aubio_midi_driver_t* new_aubio_midi_driver(aubio_settings_t* settings, 
aubio_midi_driver_t* new_aubio_midi_driver(char * name, 
    handle_midi_event_func_t handler, 
    void* event_handler_data)
{
  int i;
  aubio_midi_driver_t* driver = NULL;
  for (i = 0; aubio_midi_drivers[i].name != NULL; i++) {
    if (AUBIO_STRCMP(name,aubio_midi_drivers[i].name) == 0){
      //if (aubio_settings_str_equal(settings, "midi.driver", aubio_midi_drivers[i].name)) {
      AUBIO_DBG( "Using '%s' midi driver\n", aubio_midi_drivers[i].name);
      //driver = aubio_midi_drivers[i].new(settings, handler, event_handler_data);
      driver = aubio_midi_drivers[i].new(/*name,*/ handler, event_handler_data);
      if (driver) {
        driver->name = aubio_midi_drivers[i].name;
      }
      return driver;
    }
  }
  AUBIO_ERR("Couldn't find the requested midi driver");
  return NULL;
}

void del_aubio_midi_driver(aubio_midi_driver_t* driver)
{
  int i;

  for (i = 0; aubio_midi_drivers[i].name != NULL; i++) {
    if (aubio_midi_drivers[i].name == driver->name) {
      aubio_midi_drivers[i].free(driver);
      return;
    }
  }  
}

