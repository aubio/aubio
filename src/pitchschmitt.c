/*
   Copyright (C) 2004, 2005  Mario Lang <mlang@delysid.org>

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
#include "sample.h"
#include "pitchschmitt.h"

smpl_t aubio_schmittS16LE (aubio_pitchschmitt_t *p, uint_t nframes, signed short int *indata);

struct _aubio_pitchschmitt_t {
        uint_t blockSize;
        uint_t rate;
        signed short int *schmittBuffer;
        signed short int *schmittPointer;
};

aubio_pitchschmitt_t * new_aubio_pitchschmitt (uint_t size, uint_t samplerate)
{
  aubio_pitchschmitt_t * p = AUBIO_NEW(aubio_pitchschmitt_t);
  p->blockSize = size;
  p->schmittBuffer = AUBIO_ARRAY(signed short int,p->blockSize);
  p->schmittPointer = p->schmittBuffer;
  p->rate = samplerate;
  return p;
}

smpl_t aubio_pitchschmitt_detect (aubio_pitchschmitt_t *p, fvec_t * input)
{
  signed short int buf[input->length];
  uint_t i;
  for (i=0; i<input->length; i++) {
    buf[i] = input->data[0][i]*32768.;
  }
  return aubio_schmittS16LE(p, input->length, buf);
}

smpl_t aubio_schmittS16LE (aubio_pitchschmitt_t *p, uint_t nframes, signed short int *indata)
{
  uint_t i, j;
  uint_t blockSize = p->blockSize;
  signed short int *schmittBuffer = p->schmittBuffer;
  signed short int *schmittPointer = p->schmittPointer;

  smpl_t freq = 0., trigfact = 0.6;

  for (i=0; i<nframes; i++) {
    *schmittPointer++ = indata[i];
    if (schmittPointer-schmittBuffer >= (sint_t)blockSize) {
      sint_t endpoint, startpoint, t1, t2, A1, A2, tc, schmittTriggered;

      schmittPointer = schmittBuffer;

      for (j=0,A1=0,A2=0; j<blockSize; j++) {
	if (schmittBuffer[j]>0 && A1<schmittBuffer[j])  A1 = schmittBuffer[j];
	if (schmittBuffer[j]<0 && A2<-schmittBuffer[j]) A2 = -schmittBuffer[j];
      }
      t1 =   (sint_t)( A1 * trigfact + 0.5);
      t2 = - (sint_t)( A2 * trigfact + 0.5);
      startpoint=0;
      for (j=1; schmittBuffer[j]<=t1 && j<blockSize; j++);
      for (; !(schmittBuffer[j]  >=t2 &&
	       schmittBuffer[j+1]< t2) && j<blockSize; j++);
      startpoint=j;
      schmittTriggered=0;
      endpoint=startpoint+1;
      for(j=startpoint,tc=0; j<blockSize; j++) {
	if (!schmittTriggered) {
	  schmittTriggered = (schmittBuffer[j] >= t1);
	} else if (schmittBuffer[j]>=t2 && schmittBuffer[j+1]<t2) {
	  endpoint=j;
	  tc++;
	  schmittTriggered = 0;
	}
      }
      if (endpoint > startpoint) {
	freq = ((smpl_t)p->rate*(tc/(smpl_t)(endpoint-startpoint)));
      }
    }
  }

  p->schmittBuffer  = schmittBuffer;
  p->schmittPointer = schmittPointer;
  return freq;
}

void del_aubio_pitchschmitt (aubio_pitchschmitt_t *p)
{
  AUBIO_FREE(p->schmittBuffer);
  AUBIO_FREE(p);
}

