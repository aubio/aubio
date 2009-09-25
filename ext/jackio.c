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

#include <aubio.h>

#if HAVE_JACK
#include <jack/jack.h>
#include "aubio_priv.h"
#include "jackio.h"

typedef jack_default_audio_sample_t jack_sample_t;

#if !AUBIO_SINGLE_PRECISION
#define AUBIO_JACK_MAX_FRAMES 4096
#define AUBIO_JACK_NEEDS_CONVERSION
#endif

/**
 * jack device structure 
 */
struct _aubio_jack_t {
  /** jack client */
  jack_client_t *client;
  /** jack output ports */
  jack_port_t **oports;
  /** jack input ports */
  jack_port_t **iports;
  /** jack input buffer */
  jack_sample_t **ibufs;
  /** jack output buffer */
  jack_sample_t **obufs;
#ifdef AUBIO_JACK_NEEDS_CONVERSION 
  /** converted jack input buffer */
  smpl_t **sibufs;
  /** converted jack output buffer */
  smpl_t **sobufs;
#endif
  /** jack input channels */
  uint_t ichan;
  /** jack output channels */
  uint_t ochan;
  /** jack samplerate (Hz) */
  uint_t samplerate;
  /** jack processing function */
  aubio_process_func_t callback; 
};

/* static memory management */
static aubio_jack_t * aubio_jack_alloc(uint_t ichan, uint_t ochan);
static uint_t aubio_jack_free(aubio_jack_t * jack_setup);
/* jack callback functions */
static int aubio_jack_process(jack_nframes_t nframes, void *arg);
static void aubio_jack_shutdown (void *arg);

aubio_jack_t * new_aubio_jack(uint_t ichan, uint_t ochan, 
    aubio_process_func_t callback) {
  aubio_jack_t * jack_setup = aubio_jack_alloc (ichan, ochan);
  uint_t i;
  char * client_name = "aubio";
  char name[64];
  /* initial jack client setup */
  if ((jack_setup->client = jack_client_new (client_name)) == 0) {
    AUBIO_ERR ("jack server not running?\n");
    AUBIO_QUIT(AUBIO_FAIL);
  }

  /* set callbacks */
  jack_set_process_callback (jack_setup->client, aubio_jack_process, 
      (void*) jack_setup);
  jack_on_shutdown (jack_setup->client, aubio_jack_shutdown, 
      (void*) jack_setup);

  /* register jack output ports */
  for (i = 0; i < ochan; i++) 
  {
    AUBIO_SPRINTF(name, "out_%d", i+1);
    AUBIO_MSG("%s\n", name);
    if ((jack_setup->oports[i] = 
          jack_port_register (jack_setup->client, name, 
            JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0)) == 0) 
    {
      AUBIO_ERR("failed registering output port \"%s\"!\n", name);
      jack_client_close (jack_setup->client);
      AUBIO_QUIT(AUBIO_FAIL);
    }
  }

  /* register jack input ports */
  for (i = 0; i < ichan; i++) 
  {
    AUBIO_SPRINTF(name, "in_%d", i+1);
    AUBIO_MSG("%s\n", name);
    if ((jack_setup->iports[i] = 
          jack_port_register (jack_setup->client, name, 
            JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0)) == 0)
    {
      AUBIO_ERR("failed registering input port \"%s\"!\n", name);
      jack_client_close (jack_setup->client);
      AUBIO_QUIT(AUBIO_FAIL);
    }
  }

  /* set processing callback */
  jack_setup->callback = callback;
  return jack_setup;
}

uint_t aubio_jack_activate(aubio_jack_t *jack_setup) {
  /* get sample rate */
  jack_setup->samplerate = jack_get_sample_rate (jack_setup->client);
  /* actual jack process activation */
  if (jack_activate (jack_setup->client)) 
  {
    AUBIO_ERR("jack client activation failed");
    return 1;
  }
  return 0;
}

void aubio_jack_close(aubio_jack_t *jack_setup) {
  /* bug : should disconnect all ports first */
  jack_client_close(jack_setup->client);
  aubio_jack_free(jack_setup);
}

/* memory management */
static aubio_jack_t * aubio_jack_alloc(uint_t ichan, uint_t ochan) {
  aubio_jack_t *jack_setup = AUBIO_NEW(aubio_jack_t);
  jack_setup->ichan = ichan;
  jack_setup->ochan = ochan;
  jack_setup->oports = AUBIO_ARRAY(jack_port_t*, ichan); 
  jack_setup->iports = AUBIO_ARRAY(jack_port_t*, ochan); 
  jack_setup->ibufs  = AUBIO_ARRAY(jack_sample_t*, ichan); 
  jack_setup->obufs  = AUBIO_ARRAY(jack_sample_t*, ochan); 
#ifdef AUBIO_JACK_NEEDS_CONVERSION 
  jack_setup->sibufs = AUBIO_ARRAY(smpl_t*, ichan); 
  uint_t i;
  for (i = 0; i < ichan; i++) {
    jack_setup->sibufs[i] = AUBIO_ARRAY(smpl_t, AUBIO_JACK_MAX_FRAMES);
  }
  jack_setup->sobufs = AUBIO_ARRAY(smpl_t*, ochan); 
  for (i = 0; i < ochan; i++) {
    jack_setup->sobufs[i] = AUBIO_ARRAY(smpl_t, AUBIO_JACK_MAX_FRAMES);
  }
#endif
  return jack_setup;
}

static uint_t aubio_jack_free(aubio_jack_t * jack_setup) {
  AUBIO_FREE(jack_setup->oports);
  AUBIO_FREE(jack_setup->iports);
  AUBIO_FREE(jack_setup->ibufs );
  AUBIO_FREE(jack_setup->obufs );
  AUBIO_FREE(jack_setup);
  return AUBIO_OK;
}

/* jack callback functions */
static void aubio_jack_shutdown (void *arg UNUSED){
  AUBIO_ERR("jack shutdown\n");
  AUBIO_QUIT(AUBIO_OK);
}

static int aubio_jack_process(jack_nframes_t nframes, void *arg) {
  aubio_jack_t* dev = (aubio_jack_t *)arg;
  uint_t i;
  for (i=0;i<dev->ichan;i++) { 
    /* get readable input */
    dev->ibufs[i] = 
      (jack_sample_t *) jack_port_get_buffer (dev->iports[i], nframes);
    /* get writable output */
    dev->obufs[i] = 
      (jack_sample_t *) jack_port_get_buffer (dev->oports[i], nframes);
  }
#ifndef AUBIO_JACK_NEEDS_CONVERSION
  dev->callback(dev->ibufs,dev->obufs,nframes);
#else
  uint_t j;
  for (j = 0; j < MIN(nframes, AUBIO_JACK_MAX_FRAMES); j++) {
    for (i = 0; i < dev->ichan; i++) { 
      dev->sibufs[i][j] = (smpl_t)dev->ibufs[i][j];
    }
  }
  dev->callback(dev->sibufs, dev->sobufs, nframes);
  for (j = 0; j < MIN(nframes, AUBIO_JACK_MAX_FRAMES); j++) {
    for (i = 0; i < dev->ichan; i++) { 
      dev->obufs[i][j] = (jack_sample_t)dev->sobufs[i][j];
    }
  }
#endif
  return 0;
}


#endif /* HAVE_JACK */
