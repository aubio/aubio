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

aubio_filter_t * new_aubio_cdsgn_filter(uint_t samplerate, uint_t channels) {
  aubio_filter_t * f = new_aubio_filter(samplerate, 5, channels);
  lsmp_t * a = f->a->data[0];
  lsmp_t * b = f->b->data[0];
  /* uint_t l; */
  /* for now, 44100, cdsgn */
  a[0] =  1.000000000000000000000000000000000000000000000000000000000000; 
  a[1] = -2.134674963687040794013682898366823792457580566406250000000000; 
  a[2] =  1.279333533236063358273781886964570730924606323242187500000000; 
  a[3] = -0.149559846089396208945743182994192466139793395996093750000000; 
  a[4] =  0.004908700174624848651394604104325480875559151172637939453125; 
  b[0] =  0.217008561949218803377448239189106971025466918945312500000000;
  b[1] = -0.000000000000000222044604925031308084726333618164062500000000;
  b[2] = -0.434017123898438272888711253472138196229934692382812500000000;
  b[3] =  0.000000000000000402455846426619245903566479682922363281250000;
  b[4] =  0.217008561949218969910901932962588034570217132568359375000000;
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

