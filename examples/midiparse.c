/*
   copyright (c) 2003 paul brossier

   this program is free software; you can redistribute it and/or modify
   it under the terms of the gnu general public license as published by
   the free software foundation; either version 2 of the license, or
   (at your option) any later version.

   this program is distributed in the hope that it will be useful,
   but without any warranty; without even the implied warranty of
   merchantability or fitness for a particular purpose.  see the
   gnu general public license for more details.

   you should have received a copy of the gnu general public license
   along with this program; if not, write to the free software
   foundation, inc., 675 mass ave, cambridge, ma 02139, usa.

*/

#include "aubio.h"
#include <unistd.h>

/* not supported yet */
#ifdef LADCCA_SUPPORT
#include <ladcca/ladcca.h>
cca_client_t * aubio_cca_client;
#endif /* LADCCA_SUPPORT */

int main(int argc, char ** argv) {
#if ALSA_SUPPORT
	aubio_midi_player_t * mplay = new_aubio_midi_player();
	argc--;
	aubio_midi_player_add(mplay,argv[argc]);
	while(aubio_midi_player_play_offline(mplay) == 0)
		pause();
	aubio_midi_player_stop(mplay);
	del_aubio_midi_player(mplay);
#endif
	return 0;
} 

