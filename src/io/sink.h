/*
  Copyright (C) 2012-2013 Paul Brossier <piem@aubio.org>

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

#ifndef _AUBIO_SINK_H
#define _AUBIO_SINK_H

/** \file

  Media sink to write blocks of consecutive audio samples to file.

  \example io/test-sink.c

*/

#ifdef __cplusplus
extern "C" {
#endif

/** media sink object */
typedef struct _aubio_sink_t aubio_sink_t;

/**

  create new ::aubio_sink_t

  \param uri the file path or uri to write to
  \param samplerate sample rate to write the file at

  \return newly created ::aubio_sink_t

  Creates a new sink object.

*/
aubio_sink_t * new_aubio_sink(char_t * uri, uint_t samplerate);

/**

  write monophonic vector of length hop_size to sink

  \param s sink, created with ::new_aubio_sink
  \param write_data ::fvec_t samples to write to sink
  \param write number of frames to write

*/
void aubio_sink_do(aubio_sink_t * s, fvec_t * write_data, uint_t write);

/**

  close sink and cleanup memory

  \param s source object, created with ::new_aubio_source

*/
void del_aubio_sink(aubio_sink_t * s);

#ifdef __cplusplus
}
#endif

#endif /* _AUBIO_SINK_H */
