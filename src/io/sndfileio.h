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

#ifndef SNDFILEIO_H
#define SNDFILEIO_H

/** \file

  sndfile functions

 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * sndfile object
 */
typedef struct _aubio_sndfile_t aubio_sndfile_t;
/** 
 * Open a sound file for reading
 */
aubio_sndfile_t * new_aubio_sndfile_ro (const char * inputfile);
/**
 * Copy file model from previously opened sound file.
 */
aubio_sndfile_t * new_aubio_sndfile_wo(aubio_sndfile_t * existingfile, const char * outputname);
/** 
 * Open a sound file for writing
 */
int aubio_sndfile_open_wo (aubio_sndfile_t * file, const char * outputname);
/** 
 * Read frames data from file into an array of buffers
 */
int aubio_sndfile_read(aubio_sndfile_t * file, int frames, fvec_t ** read);
/** 
 * Read frames data from file into a single buffer
 */
int aubio_sndfile_read_mono (aubio_sndfile_t * f, int frames, fvec_t * read);
/** 
 * Write data of length frames to file
 */
int aubio_sndfile_write(aubio_sndfile_t * file, int frames, fvec_t ** write);
/**
 * Close file and delete file object
 */
int del_aubio_sndfile(aubio_sndfile_t * file);
/**
 * Return some files facts
 */
void aubio_sndfile_info(aubio_sndfile_t * file);
/**
 * Return number of channel in file
 */
uint_t aubio_sndfile_channels(aubio_sndfile_t * file);
/**
 * Return samplerate of a file (Hz)
 */
uint_t aubio_sndfile_samplerate(aubio_sndfile_t * file);

#ifdef __cplusplus
}
#endif

#endif

