
#include "aubio.h"

#ifndef JACK_SUPPORT
#define JACK_SUPPORT 0
#endif

#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isfinite */
#include "utils.h"

#ifdef LADCCA_SUPPORT
#include <ladcca/ladcca.h>
cca_client_t * aubio_cca_client;
#endif /* LADCCA_SUPPORT */


/* badly redeclare some things */
aubio_onsetdetection_type type_onset;
smpl_t threshold;
smpl_t averaging;
const char * prog_name;

void usage (FILE * stream, int exit_code)
{
	fprintf(stream, "usage: %s [ options ] \n", prog_name);
	fprintf(stream, 
			"	-j	--jack		Use Jack.\n"
			"	-o	--output	Output type.\n"
			"	-i	--input		Input type.\n"
			"	-h	--help		Display this message.\n"
			"	-v	--verbose	Print verbose message.\n"
			);
	exit(exit_code);
}

int parse_args (int argc, char **argv) {
	const char *options = "hvjo:i:O:t:a";
	int next_option;
	struct option long_options[] =
	{
		{"help", 		0, NULL, 'h'},
		{"verbose",	0, NULL, 'v'},
		{"jack", 		0, NULL, 'j'},
		{"output", 	0, NULL, 'o'},
		{"input", 	0, NULL, 'i'},
		{"onset", 	0, NULL, 'O'},
		{"threshold", 	0, NULL, 't'},
		{"averaging", 	0, NULL, 'a'},
		{NULL,			0, NULL, 0}
	};
	prog_name = argv[0];	
	if( argc < 1 ) {
		usage (stderr, 1);
		return -1;
	}
	do {
		next_option = getopt_long (argc, argv, options, 
									long_options, NULL);
		switch (next_option) {
			case 'o':
				output_filename = optarg;
				break;
			case 'i':
				input_filename = optarg;
				break;
			case 'h': 	/* help */
				usage (stdout, 0);
				return -1;
			case 'v':		/* verbose */
				verbose = 1;
				break;
			case 'j':		/* verbose */
				usejack = 1;
				break;
      case 'O':   /*onset type*/
        if (strcmp(optarg,"energy") == 0) 
          type_onset = energy;
        else if (strcmp(optarg,"specdiff") == 0) 
          type_onset = specdiff;
        else if (strcmp(optarg,"hfc") == 0) 
          type_onset = hfc;
        else if (strcmp(optarg,"complexdomain") == 0) 
          type_onset = complexdomain;
        else if (strcmp(optarg,"phase") == 0) 
          type_onset = phase;
        else {
          debug("could not get onset type.\n");
          abort();
        }
        break;
      case 't':   /* threshold value for onset */
        threshold = (smpl_t)atof(optarg);
        /*
        if (!isfinite(threshold)) {
          debug("could not get threshold.\n");
          abort();
        }
        */
        break;
      case 'a':
        averaging = 1;
        break; 
			case '?': 	/* unknown options */
				usage(stderr, 1);
        break;
			case -1: 		/* done with options */
				break;
			default: 		/*something else unexpected */
				abort ();
		}
	}
	while (next_option != -1);

	if (input_filename != NULL) {
		errmsg ("Input file : %s\n", input_filename );
	} else if (input_filename != NULL && output_filename != NULL) {
		errmsg ("Input file : %s\n", input_filename );
		errmsg ("Output file : %s\n", output_filename );
	} else {
		if (JACK_SUPPORT)
		{
			errmsg ("Jack input output\n");
			usejack = 1;
		} else {
			errmsg ("Error: Could not switch to jack mode\n   aubio was compiled without jack support\n");
			exit(1);
		}
	}	
	return 0;
}

