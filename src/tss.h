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

/** @file
 * implement Transient / Steady State separation
 */

#ifndef TSS_H
#define TSS_H

#ifdef __cplusplus
extern "C" {
#endif

/** tss publictype derived from tss.c _tss_t internal structure */
typedef struct _aubio_tss_t aubio_tss_t;

/** public creation function */
extern aubio_tss_t * new_aubio_tss(smpl_t thrs, smpl_t alfa, smpl_t beta, 
    uint_t size, uint_t overlap,uint_t channels);
/** public deletion function */
extern void del_aubio_tss(aubio_tss_t *s);

extern void aubio_tss_set_thres(aubio_tss_t *tss, smpl_t thrs);

/** split input into transient and steady states components*/
extern void aubio_tss_do(aubio_tss_t *s, cvec_t * input, cvec_t * trans, cvec_t * stead);

#ifdef __cplusplus
}
#endif

#endif /*TSS_H*/
