/*
   Copyright (C) 2003 Paul Brossier

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

/** \bug have it stop at the end of the playlist (but keep using thread, see
 *      new_aubio_timer call in midi_player)
 * 
 */

#include "aubio.h"
#include <unistd.h>

int main(int argc, char ** argv) {
#if ALSA_SUPPORT
  aubio_midi_player_t * mplay = new_aubio_midi_player();
  argc--;
  aubio_midi_player_add(mplay,argv[argc]);
  aubio_midi_player_play(mplay);
  pause();
  aubio_midi_player_stop(mplay);
  del_aubio_midi_player(mplay);
#endif
  return 0;
} 
