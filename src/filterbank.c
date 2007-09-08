/*
   Copyright (C) 2007 Amaury Hazan <ahazan@iua.upf.edu>
                  and Paul Brossier <piem@piem.org>

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

/* part of this mfcc implementation were inspired from LibXtract
   http://libxtract.sourceforge.net/
*/

#include "aubio_priv.h"
#include "filterbank.h"

/** \brief A structure to store a set of n_filters filters of lenghts win_s */
struct aubio_filterbank_t_ {
    uint_t win_s;
    uint_t n_filters;
    fvec_t *filters;
};

aubio_filterbank_t * new_aubio_filterbank(uint_t n_filters, uint_t win_s){
  /** allocating space for filterbank object */
  aubio_filterbank_t * fb = AUBIO_NEW(aubio_filterbank_t);
  uint_t filter_cnt;
  fb->win_s=win_s;
  fb->n_filters=n_filters;

  /** allocating filter tables */
  fb->filters=AUBIO_ARRAY(n_filters,fvec_t);
  for (filter_cnt=0; filter_cnt<n_filters; filter_cnt++)
    /* considering one-channel filters */
    filters[filter_cnt]=new_fvec(win_s, 1);

}

void del_aubio_filterbank(aubio_filterbank_t * fb){
  
  int filter_cnt;
  /** deleting filter tables first */
  for (filter_cnt=0; filter_cnt<fb->n_filters; filter_cnt++)
    del_fvec(fb->filters[filter_cnt]);
  AUBIO_FREE(fb->filters);
  AUBIO_FREE(fb);

}

