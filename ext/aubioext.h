/*
	 Copyright (C) 2003 Paul Brossier <piem@altern.org>

	 This program is free software; you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation; either version 2 of the License, or
	 (at your option) any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program; if not, write to the Free Software
	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
	 
*/

#ifndef __AUBIOEXT_H__
#define __AUBIOEXT_H__


#ifdef __cplusplus
extern "C" {
#endif

#include <aubio.h>
 
#ifdef JACK_SUPPORT
#include "jackio.h"
#endif 

#include "sndfileio.h"

#include "midi/midi.h"
#include "midi/midi_event.h"
#include "midi/midi_track.h"
#include "midi/midi_player.h"
#include "midi/midi_parser.h"
#include "midi/midi_file.h"
#include "midi/midi_driver.h"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
