/*
   Copyright (C) 2003-2007 Paul Brossier

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
#include "types.h"
#include "fvec.h"
#include "temporal/filter.h"
#include "temporal/filter_priv.h"
#include "temporal/adesign.h"

aubio_filter_t * new_aubio_adsgn_filter(uint_t samplerate, uint_t channels) {
  aubio_filter_t * f = new_aubio_filter(samplerate, 7, channels);
  lsmp_t * a = f->a->data[0];
  lsmp_t * b = f->b->data[0];
  /* uint_t l; */
  /* for now, 44100, adsgn */
  a[0] =  1.00000000000000000000000000000000000000000000000000000; 
  a[1] = -4.01957618111583236952810693765059113502502441406250000; 
  a[2] =  6.18940644292069386267485242569819092750549316406250000; 
  a[3] = -4.45319890354411640487342083360999822616577148437500000; 
  a[4] =  1.42084294962187751565352300531230866909027099609375000; 
  a[5] = -0.14182547383030480458998567883099894970655441284179688; 
  a[6] =  0.00435117723349511334451911181986361043527722358703613; 
  b[0] =  0.25574112520425740235907596797915175557136535644531250;
  b[1] = -0.51148225040851391653973223583307117223739624023437500;
  b[2] = -0.25574112520426162120656954357400536537170410156250000;
  b[3] =  1.02296450081703405032840237254276871681213378906250000;
  b[4] = -0.25574112520426051098354491841746494174003601074218750;
  b[5] = -0.51148225040851369449512731080176308751106262207031250;
  b[6] =  0.25574112520425729133677350546349771320819854736328125;
  /* DBG: filter coeffs at creation time */
  /*
  for (l=0; l<f->order; l++){
    AUBIO_DBG("a[%d]=\t%1.16f\tb[%d]=\t%1.16f\n",l,a[l],l,b[l]);
  }
  */
  f->a->data[0] = a;
  f->b->data[0] = b;
  return f;
}

