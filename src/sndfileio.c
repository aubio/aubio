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

#include <string.h>

#include <sndfile.h>

#include "aubio_priv.h"
#include "sample.h"
#include "sndfileio.h"
#include "mathutils.h"

#define MAX_CHANNELS 6
#define MAX_SIZE 4096

struct _aubio_file_t {
        SNDFILE *handle;
        int samplerate;
        int channels;
        int format;
        float *tmpdata; /** scratch pad for interleaving/deinterleaving. */
        int size;       /** store the size to check if realloc needed */
};

aubio_file_t * new_file_ro(const char* outputname) {
        aubio_file_t * f = AUBIO_NEW(aubio_file_t);
        SF_INFO sfinfo;
        AUBIO_MEMSET(&sfinfo, 0, sizeof (sfinfo));

        if (! (f->handle = sf_open (outputname, SFM_READ, &sfinfo))) {
                AUBIO_ERR("Not able to open input file %s.\n", outputname);
                AUBIO_ERR("%s\n",sf_strerror (NULL)); /* libsndfile err msg */
                AUBIO_QUIT(AUBIO_FAIL);
        }	

        if (sfinfo.channels > MAX_CHANNELS) { 
                AUBIO_ERR("Not able to process more than %d channels\n", MAX_CHANNELS);
                AUBIO_QUIT(AUBIO_FAIL);
        }

        f->size       = MAX_SIZE*sfinfo.channels;
        f->tmpdata    = AUBIO_ARRAY(float,f->size);
        /* get input specs */
        f->samplerate = sfinfo.samplerate;
        f->channels   = sfinfo.channels;
        f->format     = sfinfo.format;

        return f;
}

int file_open_wo(aubio_file_t * f, const char* inputname) {
        SF_INFO sfinfo;
        memset (&sfinfo, 0, sizeof (sfinfo));

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
aubio_file_t * new_file_wo(aubio_file_t * fmodel, const char *outputname) {
        aubio_file_t * f = AUBIO_NEW(aubio_file_t);
        f->samplerate    = fmodel->samplerate;
        f->channels      = fmodel->channels;
        f->format        = fmodel->format;
        file_open_wo(f, outputname);
        return f;
}


/* return 0 if properly closed, 1 otherwise */
int del_file(aubio_file_t * f) {
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
int file_read(aubio_file_t * f, int frames, fvec_t * read) {
        sf_count_t read_frames;
        int i,j, channels = f->channels;
        int nsamples = frames*channels;
        int aread;
        float *pread;	

        /* allocate data for de/interleaving reallocated when needed. */
        if (nsamples >= f->size) {
                AUBIO_ERR("Maximum file_read buffer size exceeded.");
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
                pread = fvec_get_channel(read,i);
                for (j=0; j<aread; j++) {
                        pread[j] = f->tmpdata[channels*j+i];
                }
        }
        return aread;
}

/* write 'frames' samples to file from data 
 *   return the number of frames actually written 
 */
int file_write(aubio_file_t * f, int frames, fvec_t * write) {
        sf_count_t written_frames = 0;
        int i, j,	channels = f->channels;
        int nsamples = channels*frames;
        float *pwrite;

        /* allocate data for de/interleaving reallocated when needed. */
        if (nsamples >= f->size) {
                AUBIO_ERR("Maximum file_write buffer size exceeded.");
                return -1;
                /*
                AUBIO_FREE(f->tmpdata);
                f->tmpdata = AUBIO_ARRAY(float,nsamples);
                */
        }
        //f->size = nsamples;

        /* interleaving data  */
        for (i=0; i<channels; i++) {
                pwrite = fvec_get_channel(write,i);
                for (j=0; j<frames; j++) {
                        f->tmpdata[channels*j+i] = pwrite[j];
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

uint_t aubio_file_channels(aubio_file_t * f) {
        return f->channels;
}

uint_t aubio_file_samplerate(aubio_file_t * f) {
        return f->samplerate;
}

void file_info(aubio_file_t * f) {
        AUBIO_DBG("srate    : %d\n", f->samplerate);
        AUBIO_DBG("channels : %d\n", f->channels);
        AUBIO_DBG("format   : %d\n", f->format);
}

