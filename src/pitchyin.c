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

/* This algorithm was developped by A. de Cheveigne and H. Kawahara and
 * published in:
 * 
 * de Cheveigné, A., Kawahara, H. (2002) "YIN, a fundamental frequency
 * estimator for speech and music", J. Acoust. Soc. Am. 111, 1917-1930.  
 *
 * see http://recherche.ircam.fr/equipes/pcm/pub/people/cheveign.html
 */

#include "aubio_priv.h"
#include "sample.h"
#include "mathutils.h"
#include "pitchyin.h"

/* outputs the difference function */
void aubio_pitchyin_diff(fvec_t * input, fvec_t * yin){
	uint_t c,j,tau;
	smpl_t tmp;
	for (c=0;c<input->channels;c++)
	{
		for (tau=0;tau<yin->length;tau++)
		{
			yin->data[c][tau] = 0.;
		}
		for (tau=1;tau<yin->length;tau++)
		{
			for (j=0;j<yin->length;j++)
			{
				tmp = input->data[c][j] - input->data[c][j+tau];
				yin->data[c][tau] += SQR(tmp);
			}
		}
	}
}

/* cumulative mean normalized difference function */
void aubio_pitchyin_getcum(fvec_t * yin) {
	uint_t c,tau;
	smpl_t tmp;
	for (c=0;c<yin->channels;c++)
	{
		tmp = 0.;
		yin->data[c][0] = 1.;
		//AUBIO_DBG("%f\t",yin->data[c][0]);
		for (tau=1;tau<yin->length;tau++)
		{
			tmp += yin->data[c][tau];
			yin->data[c][tau] *= tau/tmp;
			//AUBIO_DBG("%f\t",yin->data[c][tau]);
		}
		//AUBIO_DBG("\n");
	}
}

uint_t aubio_pitchyin_getpitch(fvec_t * yin) {
	uint_t c=0,tau=1;
	do 
	{
		if(yin->data[c][tau] < 0.1) { 
			while (yin->data[c][tau+1] < yin->data[c][tau]) {
				tau++;
			}
			return tau;
		}
		tau++;
	} while (tau<yin->length);
	//AUBIO_DBG("No pitch found");
	return 0;
}


/* all the above in one */
smpl_t aubio_pitchyin_getpitchfast(fvec_t * input, fvec_t * yin, smpl_t tol){
	uint_t c=0,j,tau = 0;
	sint_t period;
	smpl_t tmp = 0., tmp2 = 0.;
	yin->data[c][0] = 1.;
	for (tau=1;tau<yin->length;tau++)
	{
		yin->data[c][tau] = 0.;
		for (j=0;j<yin->length;j++)
		{
			tmp = input->data[c][j] - input->data[c][j+tau];
			yin->data[c][tau] += SQR(tmp);
		}
		tmp2 += yin->data[c][tau];
		yin->data[c][tau] *= tau/tmp2;
		period = tau-3;
		if(tau > 4 && (yin->data[c][period] < tol) && 
                                (yin->data[c][period] < yin->data[c][period+1])) {
			return vec_quadint_min(yin,period,1);
		}
        }
	return vec_quadint_min(yin,vec_min_elem(yin),1);
	//return 0;
}

