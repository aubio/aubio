
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

/* not supported yet */
#ifdef LADCCA_SUPPORT
#include <ladcca/ladcca.h>
cca_client_t * aubio_cca_client;
#endif /* LADCCA_SUPPORT */

/* settings */
const char * output_filename = NULL;
const char * input_filename  = NULL;
const char * onset_filename  = "/usr/share/sounds/aubio/woodblock.aiff";
int frames = 0;
int verbose = 0;
int usejack = 0;
int usedoubled = 1;


/* energy,specdiff,hfc,complexdomain,phase */
aubio_onsetdetection_type type_onset  = hfc;
aubio_onsetdetection_type type_onset2 = complexdomain;
smpl_t threshold                      = 0.3;
smpl_t threshold2                     = -90.;
uint_t buffer_size                    = 1024;
uint_t overlap_size                   = 512;
uint_t channels                       = 1;
uint_t samplerate                     = 44100;


aubio_file_t * file = NULL;
aubio_file_t * fileout = NULL;

aubio_pvoc_t * pv;
fvec_t * ibuf;
fvec_t * obuf;
cvec_t * fftgrain;
fvec_t * woodblock;
aubio_onsetdetection_t *o;
aubio_onsetdetection_t *o2;
fvec_t *onset;
fvec_t *onset2;
int isonset = 0;
aubio_pickpeak_t * parms;


/* pitch objects */
smpl_t pitch               = 0.;
aubio_pitchdetection_t * pitchdet;
aubio_pitchdetection_type mode = aubio_yin; // aubio_mcomb
uint_t median         = 6;

fvec_t * note_buffer  = NULL;
fvec_t * note_buffer2 = NULL;
smpl_t curlevel       = 0.;
smpl_t maxonset       = 0.;

/* midi objects */
aubio_midi_player_t * mplay; 
aubio_midi_driver_t * mdriver; 
aubio_midi_event_t  * event;

smpl_t curnote = 0.;
smpl_t newnote = 0.;
uint_t isready = 0;



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
        const char *options = "hvjo:i:O:t:s:H:a";
        int next_option;
        struct option long_options[] =
        {
                {"help"     , 0, NULL, 'h'},
                {"verbose"  , 0, NULL, 'v'},
                {"jack"     , 0, NULL, 'j'},
                {"output"   , 0, NULL, 'o'},
                {"input"    , 0, NULL, 'i'},
                {"onset"    , 0, NULL, 'O'},
                {"threshold", 0, NULL, 't'},
                {"silence"  , 0, NULL, 's'},
                {"averaging", 0, NULL, 'a'},
                {"hopsize",   0, NULL, 'H'},
                {NULL       , 0, NULL, 0}
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
                                usedoubled = 0;
                                break;
                        case 's':   /* threshold value for onset */
                                threshold2 = (smpl_t)atof(optarg);
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
                        case 'H':
                                overlap_size = atoi(optarg);
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
                debug ("Input file : %s\n", input_filename );
        } else if (input_filename != NULL && output_filename != NULL) {
                debug ("Input file : %s\n", input_filename );
                debug ("Output file : %s\n", output_filename );
        } else {
                if (JACK_SUPPORT)
                {
                        debug ("Jack input output\n");
                        usejack = 1;
                } else {
                        debug ("Error: Could not switch to jack mode\n   aubio was compiled without jack support\n");
                        exit(1);
                }
        }	
        return 0;
}

void examples_common_init(int argc,char ** argv) {


  aubio_file_t * onsetfile;
  /* parse command line arguments */
  parse_args(argc, argv);

  woodblock = new_fvec(buffer_size,1);
  if (output_filename || usejack) {
          (onsetfile = new_file_ro(onset_filename)) ||
                  (onsetfile = new_file_ro("sounds/woodblock.aiff")) ||
                  (onsetfile = new_file_ro("../sounds/woodblock.aiff"));
          /* read the output sound once */
          file_read(onsetfile, overlap_size, woodblock);
  }

  if(!usejack)
  {
    debug("Opening files ...\n");
    file = new_file_ro (input_filename);
    if (verbose) file_info(file);
    channels = aubio_file_channels(file);
    samplerate = aubio_file_samplerate(file);
    if (output_filename != NULL)
      fileout = new_file_wo(file, output_filename);
  }

  ibuf      = new_fvec(overlap_size, channels);
  obuf      = new_fvec(overlap_size, channels);
  fftgrain  = new_cvec(buffer_size, channels);

  if (usepitch) {
    pitchdet = new_aubio_pitchdetection(buffer_size*4, 
                    overlap_size, channels, samplerate, mode, aubio_freq);
  
  if (median) {
          note_buffer = new_fvec(median, 1);
          note_buffer2= new_fvec(median, 1);
  }
  }
  /* phase vocoder */
  pv = new_aubio_pvoc(buffer_size, overlap_size, channels);
  /* onsets */
  parms = new_aubio_peakpicker(threshold);
  o = new_aubio_onsetdetection(type_onset,buffer_size,channels);
  onset = new_fvec(1, channels);
  if (usedoubled)    {
    o2 = new_aubio_onsetdetection(type_onset2,buffer_size,channels);
    onset2 = new_fvec(1 , channels);
  }

}


void examples_common_del(void){
  if (usepitch) {
          send_noteon(curnote,0);
          del_aubio_pitchdetection(pitchdet);
          if (median) {
                  del_fvec(note_buffer);
                  del_fvec(note_buffer2);
          }
  }
  del_aubio_pvoc(pv);
  del_fvec(obuf);
  del_fvec(ibuf);
  del_cvec(fftgrain);
  del_fvec(onset);
}

void examples_common_process(aubio_process_func_t process_func, aubio_print_func_t print ){
  if(usejack) {
#if JACK_SUPPORT
    aubio_jack_t * jack_setup;
    debug("Jack init ...\n");
    jack_setup = new_aubio_jack(channels, channels,
          (aubio_process_func_t)process_func);
    if (usepitch) {
            debug("Midi init ...\n");
            mplay = new_aubio_midi_player();
            mdriver = new_aubio_midi_driver("alsa_seq",
                            (handle_midi_event_func_t)aubio_midi_send_event, mplay);
            event = new_aubio_midi_event();
    }
    debug("Jack activation ...\n");
    aubio_jack_activate(jack_setup);
    debug("Processing (Ctrl+C to quit) ...\n");
    pause();
    aubio_jack_close(jack_setup);
    if (usepitch) {
            send_noteon(curnote,0);
            del_aubio_midi_driver(mdriver);
    }
#else
    usage(stderr, 1);
    outmsg("Compiled without jack output, exiting.\n");
#endif

  } else {
    /* phasevoc */
    debug("Processing ...\n");

    frames = 0;

    while (overlap_size == file_read(file, overlap_size, ibuf))
    {
      isonset=0;
      process_func(ibuf->data, obuf->data, overlap_size);
      print(); 
      if (output_filename != NULL) {
        file_write(fileout,overlap_size,obuf);
      }
      frames++;
    }

    debug("Processed %d frames of %d samples.\n", frames, buffer_size);
    del_file(file);

    if (output_filename != NULL)
      del_file(fileout);

  }
}



void send_noteon(int pitch, int velo)
{
    smpl_t mpitch = (FLOOR)(freqtomidi(pitch)+.5);
    /* we should check if we use midi here, not jack */
#if ALSA_SUPPORT
    if (usejack) {
        if (velo==0) {
            aubio_midi_event_set_type(event,NOTE_OFF);
        } else {
            aubio_midi_event_set_type(event,NOTE_ON);
        }
        aubio_midi_event_set_channel(event,0);
        aubio_midi_event_set_pitch(event,mpitch);
        aubio_midi_event_set_velocity(event,velo);
        aubio_midi_direct_output(mdriver,event);
    } else 
#endif
    if (!verbose)
    {
        if (velo==0) {
            outmsg("%f\n",frames*overlap_size/(float)samplerate);
        } else {
            outmsg("%f\t%f\t", mpitch,
                        frames*overlap_size/(float)samplerate);
        }
    }
}


void note_append(fvec_t * note_buffer, smpl_t curnote) {
  uint_t i = 0;
  for (i = 0; i < note_buffer->length - 1; i++) { 
      note_buffer->data[0][i] = note_buffer->data[0][i+1];
  }
  note_buffer->data[0][note_buffer->length - 1] = curnote;
  return;
}

uint_t get_note(fvec_t *note_buffer, fvec_t *note_buffer2){
  uint_t i = 0;
  for (i = 0; i < note_buffer->length; i++) { 
      note_buffer2->data[0][i] = note_buffer->data[0][i];
  }
  return vec_median(note_buffer2);
}

