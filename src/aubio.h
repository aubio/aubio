/*
	 Copyright (C) 2003 Paul Brossier <piem@altern.org>

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

/** @mainpage 
 *
 * \section whatis All starts here ...
 *
 *	Aubio (note i need another name for this program) is a small library 
 *	for audio and control processing. The aim of this project is educative
 *	(for me, and all the others who might want to use it). The main purpose of
 *	aubio is to experiment with some bleeding-edge algorithms in a real time
 *	context. This library targets at being light and portable, and relatively
 *	fast.
 *
 *	aubio is implemented as a library of C units and functions. You can create
 *	all the C objects you need in your processing function, process those
 *	objects from a main callback function, and delete them when done.  This
 *	simple but efficient way makes it easy to write a small wrapper, for
 *	instance in the python language. (actually, GUIs should probably be build
 *	with python itself). Writing LADSPA, jmax, pd, or any other like audio
 *	plugins should be feasible too.
 *	
 *	Aubio provides various tools, some of them are listed below. I added the
 *	names of the original authors and references to corresponding articles
 *	are in the corresponding source file.
 *
 *	  - various maths tools
 *	  - phase vocoder 
 *	  - up/downsampling
 *	  - filtering (n pole/zero pairs)
 *	  - onset detection functions
 *	  - onset peak picking
 *	  - multicomb-filtering pitch detection
 *	  - transient/steady-state separation
 *	  - audio and midi devices abstractions (callback)
 *	  - audio and midi files abstractions (various access modes)
 *
 *	The midi support is kindly borrowed from the powerful Fluidsynth, written
 *	by Peter Hanappe.
 *
 *	See the README file for more information.
 *
 * \section bugs bugs and todo
 *
 *	This software is under development. It needs debugging and optimisations.
 *
 *  See <a href='bug.html'>bugs</a> and <a href='todo.html'>todo</a> lists.
 * 
 */

#ifndef AUBIO_H
#define AUBIO_H

/**
 * Global Aubio include file.
 * Programmers just need to include this file as:
 *
 * @code
 *   #include "aubio.h"
 * @endcode
 *
 * @file aubio.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/* first the generated config file */
#include "config.h"
 
/* in this order */
#include "types.h"
#include "sample.h"
#include "fft.h"
#include "phasevoc.h"
#include "mathutils.h"
#include "scale.h"
#include "hist.h"
#include "onsetdetection.h"
#include "tss.h"
#include "resample.h"

#ifdef JACK_SUPPORT
#include "jackio.h"
#endif 

#include "sndfileio.h"
#include "peakpick.h"
#include "biquad.h"
#include "filter.h"
#include "pitchdetection.h"
#include "pitchmcomb.h"
#include "pitchyin.h"

#include "midi.h"
#include "midi_event.h"
#include "midi_track.h"
#include "midi_player.h"
#include "midi_parser.h"
#include "midi_file.h"
#include "midi_driver.h"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif

