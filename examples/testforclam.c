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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <unistd.h>

#include "aubio.h"	

#include "utils.h"


const char * output_filename = NULL;
const char * input_filename = NULL;

const char * onset_filename = "/usr/share/sounds/aubio/woodblock.aiff";

int verbose = 0;
int usejack = 0;

int aubio_process(float **input, float **output, int nframes);

/* settings */

/* energy,specdiff,hfc,complexdomain,phase */
aubio_onsetdetection_type type_onset;
smpl_t threshold = 0.1;

smpl_t threshold2 = -80.;

int buffer_size = 1024;
int overlap_size = 512;
int channels = 1;
int samplerate = 44100;

/* global objects */
aubio_pvoc_t * pv;

fvec_t * ibuf;
fvec_t * obuf;
cvec_t * fftgrain;

fvec_t * woodblock;

aubio_onsetdetection_t *o;
fvec_t *onset;
//fvec_t *onset_keep;
//fvec_t *onset_proc;
int post = 5;
int pre  = 1;

int isonset = 0;

aubio_pickpeak_t * parms;

int aubio_process(float **input, float **output, int nframes) {
	unsigned int i; 	/*channels*/
	unsigned int j; 	/*frames*/   
	unsigned int pos = 0; 	/*frames%dspblocksize*/
	//unsigned int ons = 0; /*onset counter*/
	for (j=0;j<nframes;j++) {

		for (i=0;i<channels;i++) { 
			/* write input to datanew */
			fvec_write_sample(ibuf, input[i][j], i, pos);
			/* put synthnew in output */
			output[i][j] = fvec_read_sample(obuf, i, pos);
		}
		/*time for fft*/
		if (pos == overlap_size-1) { 			
			aubio_pvoc_do (pv,ibuf, fftgrain);
			aubio_onsetdetection(o,fftgrain, onset);
			fprintf(stderr,"onsetfunc: %f\n",onset->data[0][0]);
			isonset = aubio_peakpick_pimrt(onset,parms);
			if (isonset) {
				/* test for silence */
				if (aubio_silence_detection(ibuf, threshold2)==1)
					isonset=0; 
				else
					for (pos = 0; pos < overlap_size; pos++)
						obuf->data[0][pos] = 
							woodblock->data[0][pos];
			} else {
				for (pos = 0; pos < overlap_size; pos++)
					obuf->data[0][pos] = 0.;
			}
			//aubio_pvoc_rdo(pv,fftgrain, obuf);
			pos = -1; /* so it will be zero next j loop */
		}
		pos++;
	}
	return 1;
}

int main (int argc, char **argv) {  
	int frames;
	//int pos;

	aubio_file_t * file = NULL;
	aubio_file_t * fileout = NULL;

	aubio_file_t * onsetfile = new_file_ro(onset_filename);
	parse_args(argc, argv);

	if(!usejack) 
	{
		debug("Opening  files ...\n");
		file = new_file_ro (input_filename);
		file_info(file);
		channels = aubio_file_channels(file);
		if (output_filename != NULL)
			fileout = new_file_wo(file, output_filename);
	}

	ibuf = new_fvec(overlap_size, channels);
	obuf = new_fvec(overlap_size, channels);
	woodblock = new_fvec(buffer_size,1);
	fftgrain = new_cvec(buffer_size, channels);

	file_read(onsetfile, overlap_size, woodblock);

	/* phase vocoder */
	debug("Phase voc init ... \n");
	pv = new_aubio_pvoc(buffer_size, overlap_size, channels);

	/* onset init */
	debug("type_onset: %d\n", (int)type_onset);
	debug("threshold : %f\n", threshold);
	parms = new_aubio_peakpicker(threshold);

	o = new_aubio_onsetdetection(type_onset,buffer_size,channels);
	/*onset = new_fvec(overlap_size, channels);
	for (frames=0;frames<overlap_size;frames++){
		onset->data[0][frames] = 0.; 
	}*/
	onset = new_fvec(0, channels);
	
	//onset_keep = new_fvec(post+pre+1, channels);
	//onset_proc = new_fvec(post+pre+1, channels);

	// command line argument parsing
	if(usejack) {
#ifdef JACK_SUPPORT
		aubio_jack_t * jack_setup;
		debug("Jack init ...\n");
		jack_setup = new_aubio_jack(channels, channels, 
				(aubio_process_func_t)aubio_process);

		/* initialise fftgrain */
		aubio_pvoc_do (pv,ibuf, fftgrain);

		debug("Jack activation ...\n");
		aubio_jack_activate(jack_setup);

		debug("Processing (Ctrl+C to quit) ...\n");
		pause();

		aubio_jack_close(jack_setup);
#endif

	} else {
		fvec_t * inputbuf ;
		fvec_t * outputbuf;
		/* file objects */

		inputbuf 	= new_fvec(overlap_size, channels);
		outputbuf	= new_fvec(overlap_size, channels);	
		/* phasevoc init */
		debug("Processing ...\n");

		/** \bug phasevoc not initialized yet */
		/* initialise fftgrain and input bug */
		for (frames=0;frames<2;frames++) {
			aubio_pvoc_do (pv,inputbuf, fftgrain);
		}

		/** start filling the phasevoc */
		//for (frames=0;frames<1;frames++) {
			//file_read(file, overlap_size, inputbuf);
			//aubio_pvoc_do (pv,inputbuf, fftgrain);
			/** \bug onset is still badly initialized */
			//aubio_onsetdetection(o,fftgrain, onset);
		//}

		//debug("coucou\n");
		frames = 0;
		while ((overlap_size == file_read(file, overlap_size, inputbuf)) ) 
		{
			isonset=0;
			aubio_process(inputbuf->data, outputbuf->data, overlap_size);
			if (isonset) {
				/*output times in seconds*/
				//errmsg("%f\n",(frames-1)*overlap_size/(float)samplerate);
				errmsg("%f\n",(frames-3)*overlap_size/(float)samplerate);
			}
			if (output_filename != NULL)
				file_write(fileout,overlap_size,outputbuf);
			frames++;
		}
		debug("Processed %d frames of %d samples.\n", frames, buffer_size);
		del_file(file);
		if (output_filename != NULL)
			del_file(fileout);
	}

	del_aubio_pvoc(pv);

	del_fvec(obuf);
	del_fvec(ibuf);
	del_cvec(fftgrain);
	//del_fvec(onset_keep);
	//del_fvec(onset_proc);
	del_fvec(onset);

	debug("End of program.\n");
	fflush(stderr);
	return 0;
}


