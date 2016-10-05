/*
  Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

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

#ifndef AUBIO_SAMPLER_H
#define AUBIO_SAMPLER_H

/** \file

  Load and play a sound file.

  This file loads a sample and gets ready to play it.

  The `_do` function adds the new samples to the input, and write the result as
  the output.

TODO:
  - add _preset_threaded(level)
  - add _set_stretch
  - add _set_pitch

  \example synth/test-sampler.c

*/

#ifdef __cplusplus
extern "C" {
#endif

/** sampler object */
typedef struct _aubio_sampler_t aubio_sampler_t;

/** create new sampler object

  \param samplerate the sampling rate of the new sampler
  \param hop_size the block size of the new sampler

  \return the newly created ::aubio_sampler_t

*/
aubio_sampler_t * new_aubio_sampler(uint_t hop_size, uint_t samplerate);

/** load source in sampler

  \param o sampler, created by new_aubio_sampler()
  \param uri the uri of the source to load

  \return 0 if successful, non-zero otherwise

*/
uint_t aubio_sampler_load( aubio_sampler_t * o, const char_t * uri );

/** queue source in sampler

  \param o sampler, created by new_aubio_sampler()
  \param uri the uri of the source to load

  \return 0 if successfully queued, non-zero otherwise

*/
uint_t aubio_sampler_queue(aubio_sampler_t * o, const char_t * uri );

/** process sampler function

  \param o sampler, created by new_aubio_sampler()
  \param output output of the sampler

This function get new samples from the playing source into output.

*/
void aubio_sampler_do ( aubio_sampler_t * o, fvec_t * output, uint_t *read);

/** process sampler function, multiple channels

  \param o sampler, created by new_aubio_sampler()
  \param output output of the sampler

This function gets new samples from the playing source into output.

*/
void aubio_sampler_do_multi ( aubio_sampler_t * o, fmat_t * output, uint_t *read);

/** get current playing state

  \param o sampler, created by new_aubio_sampler()

  \return 0 if not playing, 1 if playing

*/
uint_t aubio_sampler_get_playing ( const aubio_sampler_t * o );

/** set current playing state

  \param o sampler, created by new_aubio_sampler()
  \param playing 0 for not playing, 1 for playing

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_set_playing ( aubio_sampler_t * o, uint_t playing );

uint_t aubio_sampler_get_loop(aubio_sampler_t * o);

/** set current looping state

  \param o sampler, created by new_aubio_sampler()
  \param looping 0 for not looping, 1 for looping

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_set_loop(aubio_sampler_t * o, uint_t loop);

/** play sample from start

  \param o sampler, created by new_aubio_sampler()

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_play ( aubio_sampler_t * o );

/** play sample from start, looping it

  \param o sampler, created by new_aubio_sampler()

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_loop ( aubio_sampler_t * o );

/** play sample from start, once

  \param o sampler, created by new_aubio_sampler()

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_trigger ( aubio_sampler_t * o );

/** stop sample

  \param o sampler, created by new_aubio_sampler()

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_stop ( aubio_sampler_t * o );

/** get end-of-file status

  \param o sampler, created by new_aubio_sampler()

  \return 1 when the eof is being reached, 0 otherwise

*/
uint_t aubio_sampler_get_eof(aubio_sampler_t * o);

/** get end-of-file status

  \param o sampler, created by new_aubio_sampler()

  \return 1 when the eof is being reached, 0 otherwise

*/
uint_t aubio_sampler_get_finished (aubio_sampler_t * o);

/** get samplerate

  \param o sampler, created by new_aubio_sampler()

  \return samplerate of the sampler

*/
uint_t aubio_sampler_get_samplerate(aubio_sampler_t * o);

/** get the number of samples that were set to zero while opening a file

  \param o sampler, created by new_aubio_sampler()
  \param waited the number of frames processed during this block

  \return the total delay in samples when the file was successfuly opened, 0
  otherwise

*/
uint_t aubio_sampler_get_waited_opening(aubio_sampler_t * o, uint_t waited);

/** seek to position

  \param o sampler, created by new_aubio_sampler()
  \param pos position to seek to, in samples

  \return 0 if successful, 1 otherwise

*/
uint_t aubio_sampler_seek(aubio_sampler_t * o, uint_t pos);

/** destroy ::aubio_sampler_t object

  \param o sampler, created by new_aubio_sampler()

*/
void del_aubio_sampler( aubio_sampler_t * o );

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_SAMPLER_H */
