/* 
 * Copyright (C) 2003  Peter Hanappe and others.
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
 * modified by Paul Brossier for aubio
 */

/**
 *
 * \bug timer still broken 
 * 	(should use alsa seq anyway) (fixed?) realtime playing is slower than
 * 	it should. moved msec_passed and deltatime to microseconds (usec)
 * 	(rounding were causing the drift) the new offline version is not quite
 * 	exact yet.
 *
 * \bug the player does not seem to understand a ``reprise'' in a file
 */

#include "aubio_priv.h"
#include "list.h"
#include "timer.h"
#include "midi.h"
#include "midi_event.h"
#include "midi_track.h"
#include "midi_player.h"
#include "midi_file.h"


/** aubio_midi_player */
struct _aubio_midi_player_t {
  aubio_track_t  *track[AUBIO_MIDI_PLAYER_MAX_TRACKS];
  aubio_timer_t* timer;
  sint_t         status;
  sint_t         loop;
  sint_t         ntracks;
  aubio_list_t*  playlist;
  char*          current_file;
  char           send_program_change;/**< should we ignore the program changes? */
  sint_t         ticks_passed;       /**< number of midi ticks that have passed */
  sint_t         usec_passed;        /**< number of microseconds that have passed */
  sint_t         miditempo;          /**< as indicated by midi settempo: n 24th of a
  *                                     usec per midi-clock. bravo! */
  lsmp_t         deltatime;          /**< microseconds per midi tick. depends on
  *                                     set-tempo */
  uint_t         division;           /**< the number of ticks per beat (quarter-note)
  *                                     in the file*/
  //aubio_synth_t*                        synth;
};

/******************************************************
 *
 *     aubio_midi_player
 */
/** new_aubio_midi_player */
aubio_midi_player_t* new_aubio_midi_player()
  //aubio_midi_player_t* new_aubio_midi_player(aubio_synth_t* synth)
{
  sint_t i;
  aubio_midi_player_t* player;
  player = AUBIO_NEW(aubio_midi_player_t);
  if (player == NULL) {
    AUBIO_ERR( "Out of memory");
    return NULL;
  }
  player->status  = AUBIO_MIDI_PLAYER_READY;
  player->loop    = 0;
  player->ntracks = 0;
  for (i = 0; i < AUBIO_MIDI_PLAYER_MAX_TRACKS; i++) {
    player->track[i] = NULL;
  }
  //player->synth = synth;
  player->timer               = NULL;
  player->playlist            = NULL;
  player->current_file        = NULL;
  player->division            = 0;
  player->send_program_change = 1;
  player->ticks_passed        = 0;
  player->usec_passed         = 0;
  player->miditempo           = 480000;
  player->deltatime           = 4000.0;
  return player;
}

/** delete_aubio_midi_player */
sint_t del_aubio_midi_player(aubio_midi_player_t* player)
{
  if (player == NULL) {
    return AUBIO_OK;
  }
  aubio_midi_player_stop(player);
  aubio_midi_player_reset(player);
  AUBIO_FREE(player);
  return AUBIO_OK;
}

/** aubio_midi_player_reset */
sint_t aubio_midi_player_reset(aubio_midi_player_t* player)
{
  sint_t i;

  for (i = 0; i < AUBIO_MIDI_PLAYER_MAX_TRACKS; i++) {
    if (player->track[i] != NULL) {
      del_aubio_track(player->track[i]);
      player->track[i] = NULL;
    }
  }
  player->current_file        = NULL;
  player->status              = AUBIO_MIDI_PLAYER_READY;
  player->loop                = 0;
  player->ntracks             = 0;
  player->division            = 0;
  player->send_program_change = 1;
  player->ticks_passed        = 0;
  player->usec_passed         = 0;
  player->miditempo           = 480000;
  player->deltatime           = 4000.0;
  return 0;
}

/** aubio_midi_player_add_track */
sint_t aubio_midi_player_add_track(aubio_midi_player_t* player, aubio_track_t* track)
{
  if (player->ntracks < AUBIO_MIDI_PLAYER_MAX_TRACKS) {
    player->track[player->ntracks++] = track;
    return AUBIO_OK;
  } else {
    return AUBIO_FAIL;
  }
}

/** aubio_midi_player_count_tracks */
sint_t aubio_midi_player_count_tracks(aubio_midi_player_t* player)
{
  return player->ntracks;
}

/** aubio_midi_player_get_track */
aubio_track_t* aubio_midi_player_get_track(aubio_midi_player_t* player, sint_t i)
{
  if ((i >= 0) && (i < AUBIO_MIDI_PLAYER_MAX_TRACKS)) {
    return player->track[i];
  } else {
    return NULL;
  }
}

/** aubio_midi_player_get_track */
sint_t aubio_midi_player_add(aubio_midi_player_t* player, char* midifile)
{
  char *s = AUBIO_STRDUP(midifile);
  player->playlist = aubio_list_append(player->playlist, s);
  return 0;
}

/** aubio_midi_player_load */
sint_t aubio_midi_player_load(aubio_midi_player_t* player, char *filename)
{
  aubio_midi_file_t* midifile;

  midifile = new_aubio_midi_file(filename); 
  if (midifile == NULL) {
    return AUBIO_FAIL;
  }
  player->division = aubio_midi_file_get_division(midifile);

  AUBIO_DBG("quarter note division=%d\n", player->division); 

  if (aubio_midi_file_load_tracks(midifile, player) != AUBIO_OK){
    return AUBIO_FAIL;
  }

  AUBIO_DBG("Tracks loaded\n"); 

  del_aubio_midi_file(midifile);
  return AUBIO_OK;  
}

/** aubio_midi_player_callback */
sint_t aubio_midi_player_callback(void* data, uint_t usec)
{
  sint_t i;
  uint_t ticks;
  uint_t delta_ticks;
  sint_t status = AUBIO_MIDI_PLAYER_DONE;
  aubio_midi_player_t* player;
  //aubio_synth_t* synth;
  player  = (aubio_midi_player_t*) data;
  //synth = player->synth;

  /* Load the next file if necessary */
  while (player->current_file == NULL) {

    if (player->playlist == NULL) {
      return 0;
    }

    aubio_midi_player_reset(player);

    player->current_file = aubio_list_get(player->playlist);
    player->playlist = aubio_list_next(player->playlist);

    //AUBIO_DBG( "%s: %d: Loading midifile %s", __FILE__, __LINE__, player->current_file);
    AUBIO_DBG("Loading midifile %s\n", player->current_file);

    if (aubio_midi_player_load(player, player->current_file) == AUBIO_OK) {

      player->ticks_passed = 0;
      player->usec_passed = 0;

      for (i = 0; i < player->ntracks; i++) {
        if (player->track[i] != NULL) {
          aubio_track_reset(player->track[i]);
        }
      }

    } else {
      player->current_file = NULL;
    }
  }

  delta_ticks = (uint_t) ((lsmp_t)(usec - player->usec_passed) / player->deltatime);
  ticks = player->ticks_passed + delta_ticks;

  for (i = 0; i < player->ntracks; i++) {
    if (!aubio_track_eot(player->track[i])) {
      status = AUBIO_MIDI_PLAYER_PLAYING;
      if (aubio_track_send_events(player->track[i], /*synth,*/ player, ticks) != AUBIO_OK) {
        /* */
      }
    }
  }

  player->status       = status;
  player->ticks_passed = ticks;
  player->usec_passed  = usec;

  if (player->status == AUBIO_MIDI_PLAYER_DONE) {
    player->current_file = NULL; 
  }

  return 1;
}

/** aubio_midi_player_play */
sint_t aubio_midi_player_play(aubio_midi_player_t* player)
{
  AUBIO_DBG("Starting midi player\n");
  if (player->status == AUBIO_MIDI_PLAYER_PLAYING) {
    AUBIO_DBG("Midi player already playing\n");
    return AUBIO_OK;
  }

  if (player->playlist == NULL) {
    AUBIO_DBG("No playlist\n");
    return AUBIO_FAIL;
  }

  player->status = AUBIO_MIDI_PLAYER_PLAYING;

  /** \bug timer is still in millisec, should be moved to microseconds,
   *     and replaced in favor of the alsa sequencer api */
  player->timer = new_aubio_timer((sint_t) player->deltatime * 1.e-3, aubio_midi_player_callback, 
      (void*) player, 1, 0);
  if (player->timer == NULL) {
    AUBIO_DBG("Failed creating timer for midi player.\n");
    return AUBIO_FAIL;
  }
  if (player->current_file == NULL) {
      AUBIO_DBG("No more file.\n");
      delete_aubio_timer(player->timer);
      return AUBIO_FAIL;
  }

  return AUBIO_OK;
}

/** aubio_midi_player_play_offline */
sint_t aubio_midi_player_play_offline(aubio_midi_player_t* player)
{
  uint_t usec = 0; /* start looking n ms in advance */
  AUBIO_DBG("Starting midi player\n");
  if (player->status == AUBIO_MIDI_PLAYER_PLAYING) {
    AUBIO_DBG("Midi player already playing\n");
    return AUBIO_OK;
  }

  if (player->playlist == NULL) {
    AUBIO_DBG("No playlist\n");
    return AUBIO_FAIL;
  }

  //AUBIO_DBG("Starting callback.\n");
  player->status = AUBIO_MIDI_PLAYER_PLAYING;

  /* no timer, no thread ! */
  while(aubio_midi_player_callback((void *)player,usec))
  { 
    /* step at least one microsecond forward */
    usec += 1 + player->deltatime;
    if (player->status == AUBIO_MIDI_PLAYER_DONE)
      break;
  }
  //AUBIO_DBG("End of callback.\n");
  
  if (player->current_file == NULL) {
      AUBIO_DBG("No more file.\n");
      return AUBIO_FAIL;
  }
  return AUBIO_OK;
}
/** aubio_midi_player_stop */
sint_t aubio_midi_player_stop(aubio_midi_player_t* player)
{
  if (player->timer != NULL) {
    delete_aubio_timer(player->timer);
  }
  player->status = AUBIO_MIDI_PLAYER_DONE;
  player->timer = NULL;
  return AUBIO_OK;
}

/** aubio_midi_player_set_loop */
sint_t aubio_midi_player_set_loop(aubio_midi_player_t* player, sint_t loop)
{
  player->loop = loop;
  return AUBIO_OK;
}

/**  aubio_midi_player_set_midi_tempo */
sint_t aubio_midi_player_set_midi_tempo(aubio_midi_player_t* player, sint_t tempo)
{
  player->miditempo = tempo;
  //player->deltatime = (lsmp_t) tempo / player->division * 1.e-3; /* in milliseconds */
  player->deltatime = (lsmp_t) tempo / player->division; /* in microseconds */

  AUBIO_DBG("Tempo Change: %d tempo=%f tick time=%f msec\n",
  //    player->usec_passed, 60.*1.e6/tempo, player->deltatime);
      player->usec_passed, 60.*1.e6/tempo, 1e-3*player->deltatime);
  
  return AUBIO_OK;
}

/** aubio_midi_player_set_bpm */
sint_t aubio_midi_player_set_bpm(aubio_midi_player_t* player, sint_t bpm)
{
  return aubio_midi_player_set_midi_tempo(player, (sint_t)((lsmp_t) 60 * 1e6 / bpm));
}

/** aubio_midi_player_join */
sint_t aubio_midi_player_join(aubio_midi_player_t* player)
{
  return player->timer? aubio_timer_join(player->timer) : AUBIO_OK;
}

/** aubio_track_send_events */
sint_t aubio_track_send_events(aubio_track_t* track, 
    //    aubio_synth_t* synth,
    aubio_midi_player_t* player,
    uint_t ticks)
{
  sint_t status = AUBIO_OK;
  aubio_midi_event_t* event;

  while (1) {

    event = track->cur;
    if (event == NULL) {
      return status;
    }
    /* prsint_t each midi tick */
    /*
       AUBIO_DBG("track=%d\tticks=%u\ttrack=%u\tdtime=%u\tnext=%u\n",
       track->num,
       ticks,
       track->ticks,
       event->dtime,
       track->ticks + event->dtime);
       */

    if (track->ticks + event->dtime > ticks) {
      return status;
    }

    track->ticks += event->dtime;
    status = aubio_midi_send_event(/*synth, */player, event);
    aubio_track_next_event(track);

  }
  return status;
}


/**
 * aubio_midi_send_event
 *
 * This is a utility function that doesn't really belong to any class or
 * structure. It is called by aubio_midi_track and aubio_midi_device.
 *
 * \note This could be moved to a callback function defined in the main programs
 */
//sint_t aubio_midi_send_event(aubio_synth_t* synth, aubio_player_t* player, aubio_midi_event_t* event)
sint_t aubio_midi_send_event(aubio_midi_player_t* player, aubio_midi_event_t* event)
{
  /* current time in seconds */
  //smpl_t print_time = player->msec_passed * 1e-3;
  smpl_t print_time = player->usec_passed * 1e-6;
  switch (event->type) {
    case NOTE_ON:
      AUBIO_MSG("Time=%f, chan=%d, pitch=%d vol=%d \n", 
          print_time, event->channel, event->param1, event->param2);
      /*if (aubio_synth_noteon(synth, event->channel, event->param1, event->param2) != AUBIO_OK) {
        return AUBIO_FAIL;
      }*/
      break;
    case NOTE_OFF:
      AUBIO_MSG("Time=%f, chan=%d, pitch=%d, vol=0\n",
          print_time, event->channel, event->param1);
      /*if (aubio_synth_noteoff(synth, event->channel, event->param1) != AUBIO_OK) {
        return AUBIO_FAIL;
      }*/
      break;
    case CONTROL_CHANGE:
      AUBIO_MSG("Time=%f Parameter, chan=%d c1=%d c2=%d\n",
          print_time, event->channel, event->param1, event->param2);
      /*if (aubio_synth_cc(synth, event->channel, event->param1, event->param2) != AUBIO_OK) {
        return AUBIO_FAIL;
      }*/
      break;
    case MIDI_SET_TEMPO:
      if (player != NULL) {
        if (aubio_midi_player_set_midi_tempo(player, event->param1) != AUBIO_OK) {
          return AUBIO_FAIL;
        }
      }
      break;
    case PROGRAM_CHANGE:
      AUBIO_MSG("Time=%f Program, chan=%d program=%d\n",
          print_time, event->channel, event->param1);
      /*if (aubio_synth_program_change(synth, event->channel, event->param1) != AUBIO_OK) {
        return AUBIO_FAIL;
      }*/
      break;
    case PITCH_BEND:
      AUBIO_MSG("Time=%f Pitchbend, chan=%d msb=%d lsb=%d \n", 
          print_time, event->channel, event->param1, event->param2);
      /*if (aubio_synth_pitch_bend(synth, event->channel, event->param1) != AUBIO_OK) {
        return AUBIO_FAIL;
      }
      break;*/
    default:
      break;
  }
  return AUBIO_OK;
}


/**
 * aubio_midi_receive_event
 *
 * \note This could be moved to a callback function defined in the main programs
 */
sint_t aubio_midi_receive_event(aubio_midi_player_t* player UNUSED, aubio_midi_event_t* event)
{
  /* current time in seconds */
  //smpl_t print_time = player->msec_passed * 1e-3;
  //smpl_t print_time = player->usec_passed * 1e-6;
  switch (event->type) {
    case NOTE_ON:
      break;
    case NOTE_OFF:
      break;
    default:
      break;
  }
  return AUBIO_OK;
}
