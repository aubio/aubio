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

#ifndef SNDFILEIO_H
#define SNDFILEIO_H

/** @file 
 * sndfile functions
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * sndfile object
 */
typedef struct _aubio_file_t aubio_file_t;
/** 
 * Open a sound file for reading
 */
aubio_file_t * new_file_ro (const char * inputfile);
/**
 * Copy file model from previously opened sound file.
 */
aubio_file_t * new_file_wo(aubio_file_t * existingfile, const char * outputname);
/** 
 * Open a sound file for writing
 */
int file_open_wo (aubio_file_t * file, const char * outputname);
/** 
 * Read frames data from file 
 */
int file_read(aubio_file_t * file, int frames, fvec_t * read);
/** 
 * Write data of length frames to file
 */
int file_write(aubio_file_t * file, int frames, fvec_t * write);
/**
 * Close file and delete file object
 */
int del_file(aubio_file_t * file);
/**
 * Return some files facts
 */
void file_info(aubio_file_t * file);
/**
 * Return number of channel in file
 */
uint_t aubio_file_channels(aubio_file_t * file);
uint_t aubio_file_samplerate(aubio_file_t * file);

#ifdef __cplusplus
}
#endif

#endif

