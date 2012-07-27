/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "config.h"

#ifdef HAVE_SNDFILE

#include <string.h>

#include <sndfile.h>

#include "aubio_priv.h"
#include "fvec.h"
#include "sndfileio.h"
#include "mathutils.h"

#define MAX_CHANNELS 6
#define MAX_SIZE 4096

struct _aubio_sndfile_t {
        SNDFILE *handle;
        int samplerate;
        int channels;
        int format;
        float *tmpdata; /** scratch pad for interleaving/deinterleaving. */
        int size;       /** store the size to check if realloc needed */
};

aubio_sndfile_t * new_aubio_sndfile_ro(const char* outputname) {
        aubio_sndfile_t * f = AUBIO_NEW(aubio_sndfile_t);
        SF_INFO sfinfo;
        AUBIO_MEMSET(&sfinfo, 0, sizeof (sfinfo));

        f->handle = sf_open (outputname, SFM_READ, &sfinfo);

        if (f->handle == NULL) {
                AUBIO_ERR("Failed opening %s: %s\n", outputname,
                        sf_strerror (NULL)); /* libsndfile err msg */
                return NULL;
        }	

        if (sfinfo.channels > MAX_CHANNELS) { 
                AUBIO_ERR("Not able to process more than %d channels\n", MAX_CHANNELS);
                return NULL;
        }

        f->size       = MAX_SIZE*sfinfo.channels;
        f->tmpdata    = AUBIO_ARRAY(float,f->size);
        /* get input specs */
        f->samplerate = sfinfo.samplerate;
        f->channels   = sfinfo.channels;
        f->format     = sfinfo.format;

        return f;
}

int aubio_sndfile_open_wo(aubio_sndfile_t * f, const char* inputname) {
        SF_INFO sfinfo;
        AUBIO_MEMSET(&sfinfo, 0, sizeof (sfinfo));

        /* define file output spec */
        sfinfo.samplerate = f->samplerate;
        sfinfo.channels   = f->channels;
        sfinfo.format     = f->format;

        if (! (f->handle = sf_open (inputname, SFM_WRITE, &sfinfo))) {
                AUBIO_ERR("Not able to open output file %s.\n", inputname);
                AUBIO_ERR("%s\n",sf_strerror (NULL)); /* libsndfile err msg */
                AUBIO_QUIT(AUBIO_FAIL);
        }	

        if (sfinfo.channels > MAX_CHANNELS) { 
                AUBIO_ERR("Not able to process more than %d channels\n", MAX_CHANNELS);
                AUBIO_QUIT(AUBIO_FAIL);
        }
        f->size       = MAX_SIZE*sfinfo.channels;
        f->tmpdata    = AUBIO_ARRAY(float,f->size);
        return AUBIO_OK;
}

/* setup file struct from existing one */
aubio_sndfile_t * new_aubio_sndfile_wo(aubio_sndfile_t * fmodel, const char *outputname) {
        aubio_sndfile_t * f = AUBIO_NEW(aubio_sndfile_t);
        f->samplerate    = fmodel->samplerate;
        f->channels      = 1; //fmodel->channels;
        f->format        = fmodel->format;
        aubio_sndfile_open_wo(f, outputname);
        return f;
}


/* return 0 if properly closed, 1 otherwise */
int del_aubio_sndfile(aubio_sndfile_t * f) {
        if (sf_close(f->handle)) {
                AUBIO_ERR("Error closing file.");
                puts (sf_strerror (NULL));
                return 1;
        }
        AUBIO_FREE(f->tmpdata);
        AUBIO_FREE(f);
        //AUBIO_DBG("File closed.\n");
        return 0;
}

/**************************************************************
 *
 * Read write methods 
 *
 */


/* read frames from file in data 
 *  return the number of frames actually read */
int aubio_sndfile_read(aubio_sndfile_t * f, int frames, fvec_t ** read) {
        sf_count_t read_frames;
        int i,j, channels = f->channels;
        int nsamples = frames*channels;
        int aread;
        smpl_t *pread;	

        /* allocate data for de/interleaving reallocated when needed. */
        if (nsamples >= f->size) {
                AUBIO_ERR("Maximum aubio_sndfile_read buffer size exceeded.");
                return -1;
                /*
                AUBIO_FREE(f->tmpdata);
                f->tmpdata = AUBIO_ARRAY(float,nsamples);
                */
        }
        //f->size = nsamples;

        /* do actual reading */
        read_frames = sf_read_float (f->handle, f->tmpdata, nsamples);

        aread = (int)FLOOR(read_frames/(float)channels);

        /* de-interleaving data  */
        for (i=0; i<channels; i++) {
                pread = (smpl_t *)fvec_get_data(read[i]);
                for (j=0; j<aread; j++) {
                        pread[j] = (smpl_t)f->tmpdata[channels*j+i];
                }
        }
        return aread;
}

int
aubio_sndfile_read_mono (aubio_sndfile_t * f, int frames, fvec_t * read)
{
  sf_count_t read_frames;
  int i, j, channels = f->channels;
  int nsamples = frames * channels;
  int aread;
  smpl_t *pread;

  /* allocate data for de/interleaving reallocated when needed. */
  if (nsamples >= f->size) {
    AUBIO_ERR ("Maximum aubio_sndfile_read buffer size exceeded.");
    return -1;
    /*
    AUBIO_FREE(f->tmpdata);
    f->tmpdata = AUBIO_ARRAY(float,nsamples);
    */
  }
  //f->size = nsamples;

  /* do actual reading */
  read_frames = sf_read_float (f->handle, f->tmpdata, nsamples);

  aread = (int) FLOOR (read_frames / (float) channels);

  /* de-interleaving data  */
  pread = (smpl_t *) fvec_get_data (read);
  for (i = 0; i < channels; i++) {
    for (j = 0; j < aread; j++) {
      pread[j] += (smpl_t) f->tmpdata[channels * j + i];
    }
  }
  for (j = 0; j < aread; j++) {
    pread[j] /= (smpl_t)channels;
  }

  return aread;
}

/* write 'frames' samples to file from data 
 *   return the number of frames actually written 
 */
int aubio_sndfile_write(aubio_sndfile_t * f, int frames, fvec_t ** write) {
        sf_count_t written_frames = 0;
        int i, j,	channels = f->channels;
        int nsamples = channels*frames;
        smpl_t *pwrite;

        /* allocate data for de/interleaving reallocated when needed. */
        if (nsamples >= f->size) {
                AUBIO_ERR("Maximum aubio_sndfile_write buffer size exceeded.");
                return -1;
                /*
                AUBIO_FREE(f->tmpdata);
                f->tmpdata = AUBIO_ARRAY(float,nsamples);
                */
        }
        //f->size = nsamples;

        /* interleaving data  */
        for (i=0; i<channels; i++) {
                pwrite = (smpl_t *)fvec_get_data(write[i]);
                for (j=0; j<frames; j++) {
                        f->tmpdata[channels*j+i] = (float)pwrite[j];
                }
        }
        written_frames = sf_write_float (f->handle, f->tmpdata, nsamples);
        return written_frames/channels;
}

/*******************************************************************
 *
 * Get object info 
 *
 */

uint_t aubio_sndfile_channels(aubio_sndfile_t * f) {
        return f->channels;
}

uint_t aubio_sndfile_samplerate(aubio_sndfile_t * f) {
        return f->samplerate;
}

void aubio_sndfile_info(aubio_sndfile_t * f) {
        AUBIO_DBG("srate    : %d\n", f->samplerate);
        AUBIO_DBG("channels : %d\n", f->channels);
        AUBIO_DBG("format   : %d\n", f->format);
}

#endif /* HAVE_SNDFILE */
