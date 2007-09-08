/*
   Copyright (C) 2006 Amaury Hazan
   Ported to aubio from LibXtract
   http://libxtract.sourceforge.net/
   

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

#define XTRACT_BARK_BANDS 26
#include "bark.c"

int xtract_init_bark(int N, float sr, int *band_limits){

    float  edges[] = {0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000}; /* Takes us up to sr = 54kHz (CCRMA: JOS)*/

    int bands = XTRACT_BARK_BANDS;
    
    while(bands--)
        band_limits[bands] = edges[bands] / sr * N;
        /*FIX shohuld use rounding, but couldn't get it to work */

    return XTRACT_SUCCESS;
}




int xtract_bark_coefficients(const float *data, const int N, const void *argv, float *result){

    int *limits, band, n;

    limits = (int *)argv;
    
    for(band = 0; band < XTRACT_BARK_BANDS - 1; band++){
        for(n = limits[band]; n < limits[band + 1]; n++)
            result[band] += data[n];
    }

    return XTRACT_SUCCESS;
}