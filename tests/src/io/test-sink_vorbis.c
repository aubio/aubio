#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

#define aubio_sink_custom "vorbis"

#ifdef HAVE_VORBISENC
// functions not exposed in the headers, declared here
typedef struct _aubio_sink_vorbis_t aubio_sink_vorbis_t;
extern aubio_sink_vorbis_t * new_aubio_sink_vorbis(const char_t *uri,
    uint_t samplerate);
extern void del_aubio_sink_vorbis (aubio_sink_vorbis_t *s);
extern uint_t aubio_sink_vorbis_open(aubio_sink_vorbis_t *s);
extern uint_t aubio_sink_vorbis_close(aubio_sink_vorbis_t *s);
extern uint_t aubio_sink_vorbis_preset_channels(aubio_sink_vorbis_t *s,
    uint_t channels);
extern uint_t aubio_sink_vorbis_preset_samplerate(aubio_sink_vorbis_t *s,
    uint_t samplerate);
extern void aubio_sink_vorbis_do(aubio_sink_vorbis_t *s, fvec_t *write_data,
    uint_t write);
extern void aubio_sink_vorbis_do_multi(aubio_sink_vorbis_t *s,
    fmat_t *write_data, uint_t write);
extern uint_t aubio_sink_vorbis_get_channels(aubio_sink_vorbis_t *s);
extern uint_t aubio_sink_vorbis_get_samplerate(aubio_sink_vorbis_t *s);

#define HAVE_AUBIO_SINK_CUSTOM
#define aubio_sink_custom_t aubio_sink_vorbis_t
#define new_aubio_sink_custom new_aubio_sink_vorbis
#define del_aubio_sink_custom del_aubio_sink_vorbis
#define aubio_sink_custom_do aubio_sink_vorbis_do
#define aubio_sink_custom_do_multi aubio_sink_vorbis_do_multi
#define aubio_sink_custom_close aubio_sink_vorbis_close
#define aubio_sink_custom_preset_samplerate aubio_sink_vorbis_preset_samplerate
#define aubio_sink_custom_preset_channels aubio_sink_vorbis_preset_channels
#define aubio_sink_custom_get_samplerate aubio_sink_vorbis_get_samplerate
#define aubio_sink_custom_get_channels aubio_sink_vorbis_get_channels
#endif /* HAVE_VORBISENC */

#include "base-sink_custom.h"

// this file uses the unstable aubio api, please use aubio_sink instead
// see src/io/sink.h and tests/src/sink/test-sink.c

int main (int argc, char **argv)
{
  return base_main(argc, argv);
}
