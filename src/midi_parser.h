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
 * midi parser
 */

#ifndef _AUBIO_MIDI_PARSER_H
#define _AUBIO_MIDI_PARSER_H


#ifdef __cplusplus
extern "C" {
#endif

/* How many parameters may a MIDI event have? */
#define AUBIO_MIDI_PARSER_MAX_PAR 3

typedef struct _aubio_midi_parser_t aubio_midi_parser_t;

aubio_midi_parser_t* new_aubio_midi_parser(void);
int del_aubio_midi_parser(aubio_midi_parser_t* parser);
aubio_midi_event_t* aubio_midi_parser_parse(aubio_midi_parser_t* parser, unsigned char c);

#ifdef __cplusplus
}
#endif

#endif /*_AUBIO_MIDI_PARSER_H*/
