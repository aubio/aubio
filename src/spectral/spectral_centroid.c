/*
   Copyright (C) 2007 Paul Brossier

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

#include "aubio_priv.h"
#include "cvec.h"
#include "spectral/spectral_centroid.h"

smpl_t aubio_spectral_centroid(cvec_t * spectrum, smpl_t samplerate) {
  uint_t i=0, j;
  smpl_t sum = 0., sc = 0.;
  for ( j = 0; j < spectrum->length; j++ ) {
    sum += spectrum->norm[i][j];
  }
  if (sum == 0.) return 0.;
  for ( j = 0; j < spectrum->length; j++ ) {
    sc += (smpl_t)j * spectrum->norm[i][j];
  }
  return sc / sum * samplerate / (smpl_t)(spectrum->length);
}


