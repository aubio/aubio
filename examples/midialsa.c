/* __copyright__ */

#include "aubio.h"
#include <unistd.h>

int main(int argc, char **argv) {
#if ALSA_SUPPORT
  aubio_midi_player_t * mplay = new_aubio_midi_player();
  aubio_midi_driver_t * mdriver = new_aubio_midi_driver("alsa_seq",
    (handle_midi_event_func_t)aubio_midi_send_event, mplay);
  pause();
  del_aubio_midi_driver(mdriver);
#endif
  return 0;
}
