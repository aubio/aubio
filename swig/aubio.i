%module aubiowrapper

%{
#include "aubio.h"
%}

/* type aliases */
typedef unsigned int uint_t;
typedef int sint_t;
typedef float smpl_t;
typedef char char_t;

/* fvec */
fvec_t * new_fvec(uint_t length);
void del_fvec(fvec_t *s);
smpl_t fvec_read_sample(fvec_t *s, uint_t position);
void fvec_write_sample(fvec_t *s, smpl_t data, uint_t position);
smpl_t * fvec_get_data(fvec_t *s);

/* cvec */
cvec_t * new_cvec(uint_t length);
void del_cvec(cvec_t *s);
void cvec_write_norm(cvec_t *s, smpl_t data, uint_t position);
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t position);
smpl_t cvec_read_norm(cvec_t *s, uint_t position);
smpl_t cvec_read_phas(cvec_t *s, uint_t position);
smpl_t * cvec_get_norm(cvec_t *s);
smpl_t * cvec_get_phas(cvec_t *s);


/* fft */
aubio_fft_t * new_aubio_fft(uint_t size);
void del_aubio_fft(aubio_fft_t * s);
void aubio_fft_do (aubio_fft_t *s, fvec_t * input, cvec_t * spectrum);
void aubio_fft_rdo (aubio_fft_t *s, cvec_t * spectrum, fvec_t * output);
void aubio_fft_do_complex (aubio_fft_t *s, fvec_t * input, fvec_t * compspec);
void aubio_fft_rdo_complex (aubio_fft_t *s, fvec_t * compspec, fvec_t * output);
void aubio_fft_get_spectrum(fvec_t * compspec, cvec_t * spectrum);
void aubio_fft_get_realimag(cvec_t * spectrum, fvec_t * compspec);
void aubio_fft_get_phas(fvec_t * compspec, cvec_t * spectrum);
void aubio_fft_get_imag(cvec_t * spectrum, fvec_t * compspec);
void aubio_fft_get_norm(fvec_t * compspec, cvec_t * spectrum);
void aubio_fft_get_real(cvec_t * spectrum, fvec_t * compspec);

/* filter */
aubio_filter_t * new_aubio_filter(uint_t order);
void aubio_filter_do(aubio_filter_t * b, fvec_t * in);
void aubio_filter_do_outplace(aubio_filter_t * b, fvec_t * in, fvec_t * out);
void aubio_filter_do_filtfilt(aubio_filter_t * b, fvec_t * in, fvec_t * tmp);
void del_aubio_filter(aubio_filter_t * b);

/* a_weighting */
aubio_filter_t * new_aubio_filter_a_weighting (uint_t samplerate);
uint_t aubio_filter_set_a_weighting (aubio_filter_t * b, uint_t samplerate);

/* c_weighting */
aubio_filter_t * new_aubio_filter_c_weighting (uint_t samplerate);
uint_t aubio_filter_set_c_weighting (aubio_filter_t * b, uint_t samplerate);

/* biquad */
aubio_filter_t * new_aubio_filter_biquad(lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);
uint_t aubio_filter_set_biquad (aubio_filter_t * b, lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);

/* mathutils */
fvec_t * new_aubio_window(char * wintype, uint_t size);
smpl_t aubio_unwrap2pi (smpl_t phase);
smpl_t aubio_bintomidi(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_miditobin(smpl_t midi, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_bintofreq(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtobin(smpl_t freq, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtomidi(smpl_t freq);
smpl_t aubio_miditofreq(smpl_t midi);
uint_t aubio_silence_detection(fvec_t * ibuf, smpl_t threshold);
smpl_t aubio_level_detection(fvec_t * ibuf, smpl_t threshold);
smpl_t aubio_zero_crossing_rate(fvec_t * input);

/* mfcc */
aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate, uint_t n_filters, uint_t n_coefs);
void del_aubio_mfcc(aubio_mfcc_t *mf);
void aubio_mfcc_do(aubio_mfcc_t *mf, cvec_t *in, fvec_t *out);

/* resampling */
#if HAVE_SAMPLERATE
aubio_resampler_t * new_aubio_resampler(float ratio, uint_t type);
void aubio_resampler_do (aubio_resampler_t *s, fvec_t * input,  fvec_t * output);
void del_aubio_resampler(aubio_resampler_t *s);
#endif /* HAVE_SAMPLERATE */

/* pvoc */
aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s);
void del_aubio_pvoc(aubio_pvoc_t *pv);
void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t *in, cvec_t * fftgrain);
void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

/* pitch detection */
aubio_pitch_t *new_aubio_pitch (char *pitch_mode,
    uint_t bufsize, uint_t hopsize, uint_t samplerate);
void aubio_pitch_do (aubio_pitch_t * p, fvec_t * ibuf, fvec_t * obuf);
uint_t aubio_pitch_set_tolerance(aubio_pitch_t *p, smpl_t thres);
uint_t aubio_pitch_set_unit(aubio_pitch_t *p, char * pitch_unit);
void del_aubio_pitch(aubio_pitch_t * p);

/* tempo */
aubio_tempo_t * new_aubio_tempo (char_t * mode,
    uint_t buf_size, uint_t hop_size, uint_t samplerate);
void aubio_tempo_do (aubio_tempo_t *o, fvec_t * input, fvec_t * tempo);
uint_t aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence);
uint_t aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold);
smpl_t aubio_tempo_get_bpm(aubio_tempo_t * bt);
smpl_t aubio_tempo_get_confidence(aubio_tempo_t * bt);
void del_aubio_tempo(aubio_tempo_t * o);

/* specdesc */
void aubio_specdesc_do (aubio_specdesc_t * o, cvec_t * fftgrain,
  fvec_t * desc);
aubio_specdesc_t *new_aubio_specdesc (char_t * method, uint_t buf_size); 
void del_aubio_specdesc (aubio_specdesc_t * o);

/* peak picker */
aubio_peakpicker_t * new_aubio_peakpicker();
void aubio_peakpicker_do(aubio_peakpicker_t * p, fvec_t * in, fvec_t * out);
fvec_t * aubio_peakpicker_get_thresholded_input(aubio_peakpicker_t * p);
void del_aubio_peakpicker(aubio_peakpicker_t * p);
uint_t aubio_peakpicker_set_threshold(aubio_peakpicker_t * p, smpl_t threshold);

/* sndfile */
%{
#include "config.h"
#if HAVE_SNDFILE
#include "sndfileio.h"
%}
aubio_sndfile_t * new_aubio_sndfile_ro (const char * inputfile);
aubio_sndfile_t * new_aubio_sndfile_wo(aubio_sndfile_t * existingfile, const char * outputname);
void aubio_sndfile_info(aubio_sndfile_t * file);
int aubio_sndfile_write(aubio_sndfile_t * file, int frames, fvec_t ** write);
int aubio_sndfile_read(aubio_sndfile_t * file, int frames, fvec_t ** read);
int aubio_sndfile_read_mono(aubio_sndfile_t * file, int frames, fvec_t * read);
int del_aubio_sndfile(aubio_sndfile_t * file);
uint_t aubio_sndfile_channels(aubio_sndfile_t * file);
uint_t aubio_sndfile_samplerate(aubio_sndfile_t * file);
%{
#endif /* HAVE_SNDFILE */
%}
