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
 * midi player
 */


#ifndef _AUBIO_MIDI_PLAYER_H
#define _AUBIO_MIDI_PLAYER_H

#define AUBIO_MIDI_PLAYER_MAX_TRACKS 128

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_midi_player_t aubio_midi_player_t;


aubio_midi_player_t* new_aubio_midi_player(void);
sint_t del_aubio_midi_player(aubio_midi_player_t* player);
sint_t aubio_midi_player_reset(aubio_midi_player_t* player);
sint_t aubio_midi_player_add_track(aubio_midi_player_t* player, aubio_track_t* track);
sint_t aubio_midi_player_count_tracks(aubio_midi_player_t* player);
aubio_track_t* aubio_midi_player_get_track(aubio_midi_player_t* player, sint_t i);
sint_t aubio_midi_player_add(aubio_midi_player_t* player, char* midifile);
sint_t aubio_midi_player_load(aubio_midi_player_t* player, char *filename);
sint_t aubio_midi_player_callback(void* data, uint_t msec);
sint_t aubio_midi_player_play(aubio_midi_player_t* player);
sint_t aubio_midi_player_play_offline(aubio_midi_player_t* player);
sint_t aubio_midi_player_stop(aubio_midi_player_t* player);
sint_t aubio_midi_player_set_loop(aubio_midi_player_t* player, sint_t loop);
sint_t aubio_midi_player_set_midi_tempo(aubio_midi_player_t* player, sint_t tempo);
sint_t aubio_midi_player_set_bpm(aubio_midi_player_t* player, sint_t bpm);
sint_t aubio_midi_player_join(aubio_midi_player_t* player);

sint_t aubio_track_send_events(aubio_track_t* track, 
/*  aubio_synth_t* synth, */
			   aubio_midi_player_t* player,
			   uint_t ticks);

sint_t aubio_midi_send_event(aubio_midi_player_t* player, aubio_midi_event_t* event);
sint_t aubio_midi_receive_event(aubio_midi_player_t* player, aubio_midi_event_t* event);

#ifdef __cplusplus
}
#endif

#endif /* _AUBIO_MIDI_PLAYER_H*/
