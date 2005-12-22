%module aubiowrapper

%{
        #include "aubio.h"
        #include "aubioext.h"
%}

#include "aubio.h"
#include "aubioext.h"

/* type aliases */
typedef unsigned int uint_t;
typedef int sint_t;
typedef float smpl_t;

/* fvec */
extern fvec_t * new_fvec(uint_t length, uint_t channels);
extern void del_fvec(fvec_t *s);
smpl_t fvec_read_sample(fvec_t *s, uint_t channel, uint_t position);
void fvec_write_sample(fvec_t *s, smpl_t data, uint_t channel, uint_t position);
smpl_t * fvec_get_channel(fvec_t *s, uint_t channel);
void fvec_put_channel(fvec_t *s, smpl_t * data, uint_t channel);
smpl_t ** fvec_get_data(fvec_t *s);

/* another way, passing -c++ option to swig */
/*
class fvec_t{
public:
    %extend {
        fvec_t(uint_t length, uint_t channels){
            return new_fvec(length, channels);
        }
        ~fvec_t() {
            del_fvec(self);
        }
        smpl_t get( uint_t channel, uint_t position) {
            return fvec_read_sample(self,channel,position);
        }
        void set( smpl_t data, uint_t channel, uint_t position) {
            fvec_write_sample(self, data, channel, position); 
        }
        #smpl_t * fvec_get_channel(fvec_t *s, uint_t channel);
        #void fvec_put_channel(fvec_t *s, smpl_t * data, uint_t channel);
    }
};
*/

/* cvec */
extern cvec_t * new_cvec(uint_t length, uint_t channels);
extern void del_cvec(cvec_t *s);


/* sndfile */
extern aubio_sndfile_t * new_aubio_sndfile_ro (const char * inputfile);
extern aubio_sndfile_t * new_aubio_sndfile_wo(aubio_sndfile_t * existingfile, const char * outputname);
extern void aubio_sndfile_info(aubio_sndfile_t * file);
extern int aubio_sndfile_write(aubio_sndfile_t * file, int frames, fvec_t * write);
extern int aubio_sndfile_read(aubio_sndfile_t * file, int frames, fvec_t * read);
extern int del_aubio_sndfile(aubio_sndfile_t * file);
extern uint_t aubio_sndfile_channels(aubio_sndfile_t * file);
extern uint_t aubio_sndfile_samplerate(aubio_sndfile_t * file);

/* fft */
extern void aubio_fft_getnorm(smpl_t * norm, fft_data_t * spectrum, uint_t size);
extern void aubio_fft_getphas(smpl_t * phase, fft_data_t * spectrum, uint_t size);

/* filter */
extern aubio_filter_t * new_aubio_filter(uint_t samplerate, uint_t order);
extern aubio_filter_t * new_aubio_adsgn_filter(uint_t samplerate);
extern aubio_filter_t * new_aubio_cdsgn_filter(uint_t samplerate);
extern void aubio_filter_do(aubio_filter_t * b, fvec_t * in);
extern void aubio_filter_do_outplace(aubio_filter_t * b, fvec_t * in, fvec_t * out);
extern void aubio_filter_do_filtfilt(aubio_filter_t * b, fvec_t * in, fvec_t * tmp);
/*extern int del_aubio_filter(aubio_filter_t * b);*/

/* biquad */
extern aubio_biquad_t * new_aubio_biquad(lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);
extern void aubio_biquad_do(aubio_biquad_t * b, fvec_t * in);
extern void aubio_biquad_do_filtfilt(aubio_biquad_t * b, fvec_t * in, fvec_t * tmp);
/*extern int del_aubio_biquad(aubio_biquad_t * b);*/

/* hist */
extern aubio_hist_t * new_aubio_hist(smpl_t flow, smpl_t fhig, uint_t nelems, uint_t channels);
extern void del_aubio_hist(aubio_hist_t *s);
extern void aubio_hist_do(aubio_hist_t *s, fvec_t * input);
extern void aubio_hist_do_notnull(aubio_hist_t *s, fvec_t * input);
extern void aubio_hist_dyn_notnull (aubio_hist_t *s, fvec_t *input);

/* mathutils */
typedef enum {
        aubio_win_rectangle,
        aubio_win_hamming,
        aubio_win_hanning,
        aubio_win_hanningz,
        aubio_win_blackman,
        aubio_win_blackman_harris,
        aubio_win_gaussian,
        aubio_win_welch,
        aubio_win_parzen
} aubio_window_type;

void aubio_window(smpl_t *w, uint_t size, aubio_window_type wintype);
smpl_t aubio_unwrap2pi (smpl_t phase);
smpl_t vec_mean(fvec_t *s);
smpl_t vec_max(fvec_t *s);
smpl_t vec_min(fvec_t *s);
uint_t vec_min_elem(fvec_t *s);
uint_t vec_max_elem(fvec_t *s);
void vec_shift(fvec_t *s);
smpl_t vec_sum(fvec_t *s);
smpl_t vec_local_energy(fvec_t * f);
smpl_t vec_local_hfc(fvec_t * f);
smpl_t vec_alpha_norm(fvec_t * DF, smpl_t alpha);
void vec_dc_removal(fvec_t * mag);
void vec_alpha_normalise(fvec_t * mag, uint_t alpha);
void vec_add(fvec_t * mag, smpl_t threshold);
void vec_adapt_thres(fvec_t * vec, fvec_t * tmp, uint_t post, uint_t pre);
smpl_t vec_moving_thres(fvec_t * vec, fvec_t * tmp, uint_t post, uint_t pre, uint_t pos);
smpl_t vec_median(fvec_t * input);
smpl_t vec_quadint(fvec_t * x,uint_t pos);
smpl_t aubio_quadfrac(smpl_t s0, smpl_t s1, smpl_t s2, smpl_t pf);
uint_t vec_peakpick(fvec_t * input, uint_t pos);
smpl_t aubio_bintomidi(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_miditobin(smpl_t midi, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_bintofreq(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtobin(smpl_t freq, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtomidi(smpl_t freq);
smpl_t aubio_miditofreq(smpl_t midi);
uint_t aubio_silence_detection(fvec_t * ibuf, smpl_t threshold);
smpl_t aubio_level_detection(fvec_t * ibuf, smpl_t threshold);

/* scale */
extern aubio_scale_t * new_aubio_scale(smpl_t flow, smpl_t fhig, smpl_t ilow, smpl_t ihig	);
extern void aubio_scale_set (aubio_scale_t *s, smpl_t ilow, smpl_t ihig, smpl_t olow, smpl_t ohig);
extern void aubio_scale_do(aubio_scale_t *s, fvec_t * input);
extern void del_aubio_scale(aubio_scale_t *s);

/* resampling */
extern aubio_resampler_t * new_aubio_resampler(float ratio, uint_t type);
extern uint_t aubio_resampler_process(aubio_resampler_t *s, fvec_t * input,  fvec_t * output);
extern void del_aubio_resampler(aubio_resampler_t *s);

/* onset detection */
typedef enum { 
        aubio_onset_energy, 
        aubio_onset_specdiff, 
        aubio_onset_hfc, 
        aubio_onset_complex, 
        aubio_onset_phase, 
        aubio_onset_kl, 
        aubio_onset_mkl 
} aubio_onsetdetection_type;
aubio_onsetdetection_t * new_aubio_onsetdetection(aubio_onsetdetection_type type, uint_t size, uint_t channels);
void aubio_onsetdetection(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_free(aubio_onsetdetection_t *o);

/* should these still be exposed ? */
void aubio_onsetdetection_energy  (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_hfc     (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_complex (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_phase   (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_specdiff(aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_kl      (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void aubio_onsetdetection_mkl     (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);

/* pvoc */
aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s, uint_t channels);
void del_aubio_pvoc(aubio_pvoc_t *pv);
void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t *in, cvec_t * fftgrain);
void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

/* pitch detection */
typedef enum {
        aubio_pitch_yin,
        aubio_pitch_mcomb,
        aubio_pitch_schmitt,
        aubio_pitch_fcomb
} aubio_pitchdetection_type;

typedef enum {
        aubio_pitchm_freq,
        aubio_pitchm_midi,
        aubio_pitchm_cent,
        aubio_pitchm_bin
} aubio_pitchdetection_mode;

smpl_t aubio_pitchdetection(aubio_pitchdetection_t * p, fvec_t * ibuf);
smpl_t aubio_pitchdetection_mcomb(aubio_pitchdetection_t *p, fvec_t * ibuf);
smpl_t aubio_pitchdetection_yin(aubio_pitchdetection_t *p, fvec_t *ibuf);

void del_aubio_pitchdetection(aubio_pitchdetection_t * p);

aubio_pitchdetection_t * new_aubio_pitchdetection(uint_t bufsize, 
		uint_t hopsize, 
		uint_t channels,
		uint_t samplerate,
		aubio_pitchdetection_type type,
		aubio_pitchdetection_mode mode);


/* pitch mcomb */
aubio_pitchmcomb_t * new_aubio_pitchmcomb(uint_t size, uint_t channels);
smpl_t aubio_pitchmcomb_detect(aubio_pitchmcomb_t * p, cvec_t * fftgrain);
uint_t aubio_pitch_cands(aubio_pitchmcomb_t * p, cvec_t * fftgrain, smpl_t * cands);
void del_aubio_pitchmcomb (aubio_pitchmcomb_t *p);

/* pitch yin */
void aubio_pitchyin_diff(fvec_t *input, fvec_t *yin);
void aubio_pitchyin_getcum(fvec_t *yin);
uint_t aubio_pitchyin_getpitch(fvec_t *yin);
uint_t aubio_pitchyin_getpitchfast(fvec_t * input, fvec_t *yin, smpl_t tol);

/* pitch schmitt */
aubio_pitchschmitt_t * new_aubio_pitchschmitt (uint_t size, uint_t samplerate);
smpl_t aubio_pitchschmitt_detect (aubio_pitchschmitt_t *p, fvec_t * input);
void del_aubio_pitchschmitt (aubio_pitchschmitt_t *p);

/* pitch fcomb */
aubio_pitchfcomb_t * new_aubio_pitchfcomb (uint_t size, uint_t samplerate);
smpl_t aubio_pitchfcomb_detect (aubio_pitchfcomb_t *p, fvec_t * input);
void del_aubio_pitchfcomb (aubio_pitchfcomb_t *p);

/* peakpicker */
aubio_pickpeak_t * new_aubio_peakpicker(smpl_t threshold);
uint_t aubio_peakpick_pimrt(fvec_t * DF, aubio_pickpeak_t * p);
smpl_t aubio_peakpick_pimrt_getval(aubio_pickpeak_t* p);
uint_t aubio_peakpick_pimrt_wt( fvec_t* DF, aubio_pickpeak_t* p, smpl_t* peakval );
void del_aubio_peakpicker(aubio_pickpeak_t * p);

/* transient/steady state separation */
aubio_tss_t * new_aubio_tss(smpl_t thrs, smpl_t alfa, smpl_t beta,
    uint_t size, uint_t overlap,uint_t channels);
void del_aubio_tss(aubio_tss_t *s);
void aubio_tss_do(aubio_tss_t *s, cvec_t * input, cvec_t * trans, cvec_t * stead);

/* beattracking */
aubio_beattracking_t * new_aubio_beattracking(uint_t winlen, uint_t channels);
void aubio_beattracking_do(aubio_beattracking_t * bt, fvec_t * dfframes, fvec_t * out);
void del_aubio_beattracking(aubio_beattracking_t * p);



/* jack */
#ifdef JACK_SUPPORT
extern aubio_jack_t * new_aubio_jack (uint_t inchannels, uint_t outchannels, aubio_process_func_t callback); 
typedef int (*aubio_process_func_t)(smpl_t **input, smpl_t **output, int nframes);
extern uint_t aubio_jack_activate(aubio_jack_t *jack_setup);
extern void aubio_jack_close(aubio_jack_t *jack_setup);
#endif 

/* midi */
enum aubio_midi_event_type {
  /* channel messages */
  NOTE_OFF = 0x80,
  NOTE_ON = 0x90,
  KEY_PRESSURE = 0xa0,
  CONTROL_CHANGE = 0xb0,
  PROGRAM_CHANGE = 0xc0,
  CHANNEL_PRESSURE = 0xd0,
  PITCH_BEND = 0xe0,
  /* system exclusive */
  MIDI_SYSEX = 0xf0,
  /* system common - never in midi files */
  MIDI_TIME_CODE = 0xf1,
  MIDI_SONG_POSITION = 0xf2,
  MIDI_SONG_SELECT = 0xf3,
  MIDI_TUNE_REQUEST = 0xf6,
  MIDI_EOX = 0xf7,
  /* system real-time - never in midi files */
  MIDI_SYNC = 0xf8,
  MIDI_TICK = 0xf9,
  MIDI_START = 0xfa,
  MIDI_CONTINUE = 0xfb,
  MIDI_STOP = 0xfc,
  MIDI_ACTIVE_SENSING = 0xfe,
  MIDI_SYSTEM_RESET = 0xff,
  /* meta event - for midi files only */
  MIDI_META_EVENT = 0xff
};

enum aubio_midi_control_change {
  BANK_SELECT_MSB = 0x00,
  MODULATION_MSB = 0x01,
  BREATH_MSB = 0x02,
  FOOT_MSB = 0x04,
  PORTAMENTO_TIME_MSB = 0x05,
  DATA_ENTRY_MSB = 0x06,
  VOLUME_MSB = 0x07,
  BALANCE_MSB = 0x08,
  PAN_MSB = 0x0A,
  EXPRESSION_MSB = 0x0B,
  EFFECTS1_MSB = 0x0C,
  EFFECTS2_MSB = 0x0D,
  GPC1_MSB = 0x10, /* general purpose controller */
  GPC2_MSB = 0x11,
  GPC3_MSB = 0x12,
  GPC4_MSB = 0x13,
  BANK_SELECT_LSB = 0x20,
  MODULATION_WHEEL_LSB = 0x21,
  BREATH_LSB = 0x22,
  FOOT_LSB = 0x24,
  PORTAMENTO_TIME_LSB = 0x25,
  DATA_ENTRY_LSB = 0x26,
  VOLUME_LSB = 0x27,
  BALANCE_LSB = 0x28,
  PAN_LSB = 0x2A,
  EXPRESSION_LSB = 0x2B,
  EFFECTS1_LSB = 0x2C,
  EFFECTS2_LSB = 0x2D,
  GPC1_LSB = 0x30,
  GPC2_LSB = 0x31,
  GPC3_LSB = 0x32,
  GPC4_LSB = 0x33,
  SUSTAIN_SWITCH = 0x40,
  PORTAMENTO_SWITCH = 0x41,
  SOSTENUTO_SWITCH = 0x42,
  SOFT_PEDAL_SWITCH = 0x43,
  LEGATO_SWITCH = 0x45,
  HOLD2_SWITCH = 0x45,
  SOUND_CTRL1 = 0x46,
  SOUND_CTRL2 = 0x47,
  SOUND_CTRL3 = 0x48,
  SOUND_CTRL4 = 0x49,
  SOUND_CTRL5 = 0x4A,
  SOUND_CTRL6 = 0x4B,
  SOUND_CTRL7 = 0x4C,
  SOUND_CTRL8 = 0x4D,
  SOUND_CTRL9 = 0x4E,
  SOUND_CTRL10 = 0x4F,
  GPC5 = 0x50,
  GPC6 = 0x51,
  GPC7 = 0x52,
  GPC8 = 0x53,
  PORTAMENTO_CTRL = 0x54,
  EFFECTS_DEPTH1 = 0x5B,
  EFFECTS_DEPTH2 = 0x5C,
  EFFECTS_DEPTH3 = 0x5D,
  EFFECTS_DEPTH4 = 0x5E,
  EFFECTS_DEPTH5 = 0x5F,
  DATA_ENTRY_INCR = 0x60,
  DATA_ENTRY_DECR = 0x61,
  NRPN_LSB = 0x62,
  NRPN_MSB = 0x63,
  RPN_LSB = 0x64,
  RPN_MSB = 0x65,
  ALL_SOUND_OFF = 0x78,
  ALL_CTRL_OFF = 0x79,
  LOCAL_CONTROL = 0x7A,
  ALL_NOTES_OFF = 0x7B,
  OMNI_OFF = 0x7C,
  OMNI_ON = 0x7D,
  POLY_OFF = 0x7E,
  POLY_ON = 0x7F
};

enum midi_meta_event {
  MIDI_COPYRIGHT = 0x02,
  MIDI_TRACK_NAME = 0x03,
  MIDI_INST_NAME = 0x04,
  MIDI_LYRIC = 0x05,
  MIDI_MARKER = 0x06,
  MIDI_CUE_POINT = 0x07,
  MIDI_EOT = 0x2f,
  MIDI_SET_TEMPO = 0x51,
  MIDI_SMPTE_OFFSET = 0x54,
  MIDI_TIME_SIGNATURE = 0x58,
  MIDI_KEY_SIGNATURE = 0x59,
  MIDI_SEQUENCER_EVENT = 0x7f
};

enum aubio_player_status 
{
  AUBIO_MIDI_PLAYER_READY,
  AUBIO_MIDI_PLAYER_PLAYING,
  AUBIO_MIDI_PLAYER_DONE
};

enum aubio_driver_status 
{
  AUBIO_MIDI_READY,
  AUBIO_MIDI_LISTENING,
  AUBIO_MIDI_DONE
};

/* midi event */
aubio_midi_event_t* new_aubio_midi_event(void);
int del_aubio_midi_event(aubio_midi_event_t* event);
int aubio_midi_event_set_type(aubio_midi_event_t* evt, int type);
int aubio_midi_event_get_type(aubio_midi_event_t* evt);
int aubio_midi_event_set_channel(aubio_midi_event_t* evt, int chan);
int aubio_midi_event_get_channel(aubio_midi_event_t* evt);
int aubio_midi_event_get_key(aubio_midi_event_t* evt);
int aubio_midi_event_set_key(aubio_midi_event_t* evt, int key);
int aubio_midi_event_get_velocity(aubio_midi_event_t* evt);
int aubio_midi_event_set_velocity(aubio_midi_event_t* evt, int vel);
int aubio_midi_event_get_control(aubio_midi_event_t* evt);
int aubio_midi_event_set_control(aubio_midi_event_t* evt, int ctrl);
int aubio_midi_event_get_value(aubio_midi_event_t* evt);
int aubio_midi_event_set_value(aubio_midi_event_t* evt, int val);
int aubio_midi_event_get_program(aubio_midi_event_t* evt);
int aubio_midi_event_set_program(aubio_midi_event_t* evt, int val);
int aubio_midi_event_get_pitch(aubio_midi_event_t* evt);
int aubio_midi_event_set_pitch(aubio_midi_event_t* evt, int val);
int aubio_midi_event_length(unsigned char status);

/* midi track */
aubio_track_t* new_aubio_track(int num);
int del_aubio_track(aubio_track_t* track);
int aubio_track_set_name(aubio_track_t* track, char* name);
char* aubio_track_get_name(aubio_track_t* track);
int aubio_track_add_event(aubio_track_t* track, aubio_midi_event_t* evt);
aubio_midi_event_t* aubio_track_first_event(aubio_track_t* track);
aubio_midi_event_t* aubio_track_next_event(aubio_track_t* track);
int aubio_track_get_duration(aubio_track_t* track);
int aubio_track_reset(aubio_track_t* track);
int aubio_track_count_events(aubio_track_t* track, int* on, int* off);

/* midi player */
aubio_midi_player_t* new_aubio_midi_player(void);
sint_t del_aubio_midi_player(aubio_midi_player_t* player);
sint_t aubio_midi_player_reset(aubio_midi_player_t* player);
sint_t aubio_midi_player_add_track(aubio_midi_player_t* player, aubio_track_t* track);
sint_t aubio_midi_player_count_tracks(aubio_midi_player_t* player);
aubio_track_t* aubio_midi_player_get_track(aubio_midi_player_t* player, sint_t i);
sint_t aubio_midi_player_add(aubio_midi_player_t* player, char* midifile);
sint_t aubio_midi_player_load(aubio_midi_player_t* player, char *filename);
sint_t aubio_midi_player_callback(void* data, uint_t msec);
sint_t aubio_midi_player_play(aubio_midi_player_t* player);
sint_t aubio_midi_player_play_offline(aubio_midi_player_t* player);
sint_t aubio_midi_player_stop(aubio_midi_player_t* player);
sint_t aubio_midi_player_set_loop(aubio_midi_player_t* player, sint_t loop);
sint_t aubio_midi_player_set_midi_tempo(aubio_midi_player_t* player, sint_t tempo);
sint_t aubio_midi_player_set_bpm(aubio_midi_player_t* player, sint_t bpm);
sint_t aubio_midi_player_join(aubio_midi_player_t* player);
sint_t aubio_track_send_events(aubio_track_t* track, 
/*  aubio_synth_t* synth, */
			   aubio_midi_player_t* player,
			   uint_t ticks);
sint_t aubio_midi_send_event(aubio_midi_player_t* player, aubio_midi_event_t* event);

/* midi parser */
aubio_midi_parser_t* new_aubio_midi_parser(void);
int del_aubio_midi_parser(aubio_midi_parser_t* parser);
aubio_midi_event_t* aubio_midi_parser_parse(aubio_midi_parser_t* parser, unsigned char c);

/* midi file */
aubio_midi_file_t* new_aubio_midi_file(char* filename);
void del_aubio_midi_file(aubio_midi_file_t* mf);
int aubio_midi_file_read_mthd(aubio_midi_file_t* midifile);
int aubio_midi_file_load_tracks(aubio_midi_file_t* midifile, aubio_midi_player_t* player);
int aubio_midi_file_read_track(aubio_midi_file_t* mf, aubio_midi_player_t* player, int num);
int aubio_midi_file_read_event(aubio_midi_file_t* mf, aubio_track_t* track);
int aubio_midi_file_read_varlen(aubio_midi_file_t* mf);
int aubio_midi_file_getc(aubio_midi_file_t* mf);
int aubio_midi_file_push(aubio_midi_file_t* mf, int c);
int aubio_midi_file_read(aubio_midi_file_t* mf, void* buf, int len);
int aubio_midi_file_skip(aubio_midi_file_t* mf, int len);
int aubio_midi_file_read_tracklen(aubio_midi_file_t* mf);
int aubio_midi_file_eot(aubio_midi_file_t* mf);
int aubio_midi_file_get_division(aubio_midi_file_t* midifile);


/* midi driver */
aubio_midi_driver_t* new_aubio_midi_driver(char * name,
        handle_midi_event_func_t handler, void* event_handler_data);
typedef int* (handle_midi_event_func_t) (void* data, aubio_midi_event_t* event);
void del_aubio_midi_driver(aubio_midi_driver_t* driver);
void aubio_midi_driver_settings(aubio_settings_t* settings);

/* timer */
/*
extern aubio_timer_t* new_aubio_timer(int msec, int * callback, 
                        void* data, int new_thread, int auto_destroy);
extern int aubio_timer_join(aubio_timer_t* timer);
extern int aubio_timer_stop(aubio_timer_t* timer);
extern int delete_aubio_timer(aubio_timer_t* timer);
extern void * aubio_timer_start(void * data);
extern void aubio_time_config(void);
*/

/* list */
/*
extern struct aubio_list_t* new_aubio_list(void);
extern void del_aubio_list(struct aubio_list_t *list);
extern void del_aubio_list1(struct aubio_list_t *list);
#extern struct aubio_list_t* aubio_list_sort(struct aubio_list_t *list, aubio_compare_func_t compare_func);
extern struct aubio_list_t* aubio_list_append(struct aubio_list_t *list, void* data);
extern struct aubio_list_t* aubio_list_prepend(struct aubio_list_t *list, void* data);
extern struct aubio_list_t* aubio_list_remove(struct aubio_list_t *list, void* data);
extern struct aubio_list_t* aubio_list_remove_link(struct aubio_list_t *list, struct aubio_list_t *llink);
extern struct aubio_list_t* aubio_list_nth(struct aubio_list_t *list, int n);
extern struct aubio_list_t* aubio_list_last(struct aubio_list_t *list);
extern struct aubio_list_t* aubio_list_insert_at(struct aubio_list_t *list, int n, void* data);
*/
