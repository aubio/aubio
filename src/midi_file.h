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
 * midi file reader
 */

#ifndef _AUBIO_MIDI_FILE_H
#define _AUBIO_MIDI_FILE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_midi_file_t aubio_midi_file_t;


aubio_midi_file_t* new_aubio_midi_file(char* filename);
void del_aubio_midi_file(aubio_midi_file_t* mf);
int aubio_midi_file_read_mthd(aubio_midi_file_t* midifile);
int aubio_midi_file_load_tracks(aubio_midi_file_t* midifile, aubio_midi_player_t* player);
int aubio_midi_file_read_track(aubio_midi_file_t* mf, aubio_midi_player_t* player, int num);
int aubio_midi_file_read_event(aubio_midi_file_t* mf, aubio_track_t* track);
int aubio_midi_file_read_varlen(aubio_midi_file_t* mf);
int aubio_midi_file_getc(aubio_midi_file_t* mf);
int aubio_midi_file_push(aubio_midi_file_t* mf, int c);
int aubio_midi_file_read(aubio_midi_file_t* mf, void* buf, int len);
int aubio_midi_file_skip(aubio_midi_file_t* mf, int len);
int aubio_midi_file_read_tracklen(aubio_midi_file_t* mf);
int aubio_midi_file_eot(aubio_midi_file_t* mf);
int aubio_midi_file_get_division(aubio_midi_file_t* midifile);


/* From ctype.h */
#define aubio_isascii(c)    (((c) & ~0x7f) == 0)  
int aubio_isasciistring(char* s);
long aubio_getlength(unsigned char *s);

#ifdef __cplusplus
}
#endif

#endif /*_AUBIO_MIDI_FILE_H*/
