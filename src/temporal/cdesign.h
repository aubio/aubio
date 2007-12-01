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

#include "temporal/filter.h"

/** create a new C-design filter 

  \param samplerate sampling-rate of the signal to filter 

*/
aubio_filter_t * new_aubio_cdsgn_filter(uint_t samplerate);

#define aubio_cdsgn_filter_do aubio_filter_do
#define del_aubio_cdsgn_filter del_aubio_filter
