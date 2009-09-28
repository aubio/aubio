/*
  Copyright (C) 2009 Paul Brossier <piem@aubio.org>

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

/** @file
 *  various utilities functions for fvec and cvec objects
 *
 */

#ifndef _VECUTILS_H
#define _VECUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#define AUBIO_OP_PROTO(OPNAME, TYPE) \
void TYPE ## _ ## OPNAME (TYPE ## _t *o);

#define AUBIO_OP_C_AND_F_PROTO(OPNAME) \
  AUBIO_OP_PROTO(OPNAME, fvec) \
  AUBIO_OP_PROTO(OPNAME, cvec)

AUBIO_OP_C_AND_F_PROTO(exp)
AUBIO_OP_C_AND_F_PROTO(cos)
AUBIO_OP_C_AND_F_PROTO(sin)
AUBIO_OP_C_AND_F_PROTO(abs)
//AUBIO_OP_C_AND_F_PROTO(pow)
AUBIO_OP_C_AND_F_PROTO(sqrt)
AUBIO_OP_C_AND_F_PROTO(log10)
AUBIO_OP_C_AND_F_PROTO(log)
AUBIO_OP_C_AND_F_PROTO(floor)
AUBIO_OP_C_AND_F_PROTO(ceil)
AUBIO_OP_C_AND_F_PROTO(round)

/** raise each vector elements to the power pow

  \param s vector to modify
  \param pow power to raise to

*/
void fvec_pow (fvec_t *s, smpl_t pow);

//void fvec_log10 (fvec_t *s);

#ifdef __cplusplus
}
#endif

#endif /*_VECUTILS_H*/
