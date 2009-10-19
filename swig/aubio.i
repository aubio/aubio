%module aubiowrapper

%{
#include "aubio.h"
%}

/* type aliases */
typedef unsigned int uint_t;
typedef int sint_t;
typedef float smpl_t;

/* fvec */
fvec_t * new_fvec(uint_t length, uint_t channels);
void del_fvec(fvec_t *s);
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
cvec_t * new_cvec(uint_t length, uint_t channels);
void del_cvec(cvec_t *s);
void cvec_write_norm(cvec_t *s, smpl_t data, uint_t channel, uint_t position);
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t channel, uint_t position);
smpl_t cvec_read_norm(cvec_t *s, uint_t channel, uint_t position);
smpl_t cvec_read_phas(cvec_t *s, uint_t channel, uint_t position);
void cvec_put_norm_channel(cvec_t *s, smpl_t * data, uint_t channel);
void cvec_put_phas_channel(cvec_t *s, smpl_t * data, uint_t channel);
smpl_t * cvec_get_norm_channel(cvec_t *s, uint_t channel);
smpl_t * cvec_get_phas_channel(cvec_t *s, uint_t channel);
smpl_t ** cvec_get_norm(cvec_t *s);
smpl_t ** cvec_get_phas(cvec_t *s);


/* fft */
aubio_fft_t * new_aubio_fft(uint_t size, uint_t channels);
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
aubio_filter_t * new_aubio_filter(uint_t order, uint_t channels);
void aubio_filter_do(aubio_filter_t * b, fvec_t * in);
void aubio_filter_do_outplace(aubio_filter_t * b, fvec_t * in, fvec_t * out);
void aubio_filter_do_filtfilt(aubio_filter_t * b, fvec_t * in, fvec_t * tmp);
void del_aubio_filter(aubio_filter_t * b);

aubio_filter_t * new_aubio_filter_a_weighting (uint_t channels, uint_t samplerate);
uint_t aubio_filter_set_a_weighting (aubio_filter_t * b, uint_t samplerate);

aubio_filter_t * new_aubio_filter_c_weighting (uint_t channels, uint_t samplerate);
uint_t aubio_filter_set_c_weighting (aubio_filter_t * b, uint_t samplerate);

/* biquad */
aubio_filter_t * new_aubio_filter_biquad(lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3, uint_t channels);
uint_t aubio_filter_set_biquad (aubio_filter_t * b, lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);

/* hist */
aubio_hist_t * new_aubio_hist(smpl_t flow, smpl_t fhig, uint_t nelems, uint_t channels);
void del_aubio_hist(aubio_hist_t *s);
void aubio_hist_do(aubio_hist_t *s, fvec_t * input);
void aubio_hist_do_notnull(aubio_hist_t *s, fvec_t * input);
void aubio_hist_dyn_notnull(aubio_hist_t *s, fvec_t *input);
void aubio_hist_weight(aubio_hist_t *s);
smpl_t aubio_hist_mean(aubio_hist_t *s);

/* mathutils */
fvec_t * new_aubio_window(char * wintype, uint_t size);
smpl_t aubio_unwrap2pi (smpl_t phase);
smpl_t fvec_mean(fvec_t *s);
smpl_t fvec_max(fvec_t *s);
smpl_t fvec_min(fvec_t *s);
uint_t fvec_min_elem(fvec_t *s);
uint_t fvec_max_elem(fvec_t *s);
void fvec_shift(fvec_t *s);
smpl_t fvec_sum(fvec_t *s);
smpl_t fvec_local_energy(fvec_t * f);
smpl_t fvec_local_hfc(fvec_t * f);
smpl_t fvec_alpha_norm(fvec_t * DF, smpl_t alpha);
void fvec_min_removal(fvec_t * mag);
void fvec_alpha_normalise(fvec_t * mag, uint_t alpha);
void fvec_add(fvec_t * mag, smpl_t threshold);
void fvec_adapt_thres(fvec_t * vec, fvec_t * tmp, uint_t post, uint_t pre, uint_t channel);
smpl_t fvec_moving_thres(fvec_t * vec, fvec_t * tmp, uint_t post, uint_t pre, uint_t pos, uint_t channel);
smpl_t fvec_median_channel (fvec_t * input, uint_t channel);
smpl_t fvec_quadint(fvec_t * x,uint_t pos, uint_t span);
smpl_t aubio_quadfrac(smpl_t s0, smpl_t s1, smpl_t s2, smpl_t pf);
uint_t fvec_peakpick(fvec_t * input, uint_t pos);
smpl_t aubio_bintomidi(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_miditobin(smpl_t midi, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_bintofreq(smpl_t bin, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtobin(smpl_t freq, smpl_t samplerate, smpl_t fftsize);
smpl_t aubio_freqtomidi(smpl_t freq);
smpl_t aubio_miditofreq(smpl_t midi);
uint_t aubio_silence_detection(fvec_t * ibuf, smpl_t threshold);
smpl_t aubio_level_detection(fvec_t * ibuf, smpl_t threshold);
void aubio_autocorr(fvec_t * in, fvec_t * acf);
smpl_t aubio_zero_crossing_rate(fvec_t * input);
smpl_t aubio_spectral_centroid(cvec_t * spectrum, smpl_t samplerate);

/* filterbank */
aubio_filterbank_t * new_aubio_filterbank(uint_t win_s, uint_t channels);
void aubio_filterbank_set_triangle_bands (aubio_filterbank_t *fb, fvec_t *freqs, uint_t samplerate);
void aubio_filterbank_set_mel_coeffs_slaney(aubio_filterbank_t *fb, uint_t samplerate);
void del_aubio_filterbank(aubio_filterbank_t * fb);
void aubio_filterbank_do(aubio_filterbank_t * fb, cvec_t * in, fvec_t *out);
fvec_t * aubio_filterbank_get_coeffs(aubio_filterbank_t * fb);

/* mfcc */
aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate, uint_t n_filters, uint_t n_coefs);
void del_aubio_mfcc(aubio_mfcc_t *mf);
void aubio_mfcc_do(aubio_mfcc_t *mf, cvec_t *in, fvec_t *out);

/* scale */
aubio_scale_t * new_aubio_scale(smpl_t flow, smpl_t fhig, smpl_t ilow, smpl_t ihig);
uint_t aubio_scale_set_limits (aubio_scale_t *s, smpl_t ilow, smpl_t ihig, smpl_t olow, smpl_t ohig);
void aubio_scale_do(aubio_scale_t *s, fvec_t * input);
void del_aubio_scale(aubio_scale_t *s);

/* resampling */
#if HAVE_SAMPLERATE
aubio_resampler_t * new_aubio_resampler(float ratio, uint_t type);
void aubio_resampler_do (aubio_resampler_t *s, fvec_t * input,  fvec_t * output);
void del_aubio_resampler(aubio_resampler_t *s);
#endif /* HAVE_SAMPLERATE */

/* onset detection */
aubio_onsetdetection_t * new_aubio_onsetdetection(char * onset_mode, uint_t size, uint_t channels);
void aubio_onsetdetection_do (aubio_onsetdetection_t *o, cvec_t * fftgrain, fvec_t * onset);
void del_aubio_onsetdetection(aubio_onsetdetection_t *o);

/* pvoc */
aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s, uint_t channels);
void del_aubio_pvoc(aubio_pvoc_t *pv);
void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t *in, cvec_t * fftgrain);
void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

/* pitch detection */
aubio_pitch_t *new_aubio_pitch (char *pitch_mode,
    uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate);
void aubio_pitch_do (aubio_pitch_t * p, fvec_t * ibuf, fvec_t * obuf);
uint_t aubio_pitch_set_tolerance(aubio_pitch_t *p, smpl_t thres);
uint_t aubio_pitch_set_unit(aubio_pitch_t *p, char * pitch_unit);
void del_aubio_pitch(aubio_pitch_t * p);

/* pitch mcomb */
aubio_pitchmcomb_t * new_aubio_pitchmcomb(uint_t bufsize, uint_t hopsize, uint_t channels);
void aubio_pitchmcomb_do (aubio_pitchmcomb_t * p, cvec_t * fftgrain, fvec_t * out);
void del_aubio_pitchmcomb (aubio_pitchmcomb_t *p);

/* pitch yin */
aubio_pitchyin_t * new_aubio_pitchyin(uint_t bufsize);
void aubio_pitchyin_do (aubio_pitchyin_t *o, fvec_t * in, fvec_t *out);
void del_aubio_pitchyin (aubio_pitchyin_t *p);

/* pitch schmitt */
aubio_pitchschmitt_t * new_aubio_pitchschmitt (uint_t size);
void aubio_pitchschmitt_do (aubio_pitchschmitt_t *p, fvec_t * input, fvec_t *out);
void del_aubio_pitchschmitt (aubio_pitchschmitt_t *p);

/* pitch fcomb */
aubio_pitchfcomb_t * new_aubio_pitchfcomb (uint_t size, uint_t hopsize, uint_t samplerate);
void aubio_pitchfcomb_do (aubio_pitchfcomb_t *p, fvec_t * input, fvec_t *out);
void del_aubio_pitchfcomb (aubio_pitchfcomb_t *p);

/* peakpicker */
aubio_peakpicker_t * new_aubio_peakpicker(uint_t channels);
void aubio_peakpicker_do(aubio_peakpicker_t * p, fvec_t * df, fvec_t * out);
smpl_t aubio_peakpicker_get_thresholded_input(aubio_peakpicker_t* p);
void del_aubio_peakpicker(aubio_peakpicker_t * p);
uint_t aubio_peakpicker_set_threshold(aubio_peakpicker_t * p, smpl_t threshold);
smpl_t aubio_peakpicker_get_threshold(aubio_peakpicker_t * p);

/* transient/steady state separation */
aubio_tss_t *new_aubio_tss (uint_t win_s, uint_t hop_s, uint_t channels);
void del_aubio_tss (aubio_tss_t * s);
void aubio_tss_do (aubio_tss_t * s, cvec_t * input, cvec_t * trans,
    cvec_t * stead);
uint_t aubio_tss_set_threshold (aubio_tss_t * o, smpl_t thrs);
uint_t aubio_tss_set_alpha (aubio_tss_t * o, smpl_t alpha);
uint_t aubio_tss_set_beta (aubio_tss_t * o, smpl_t beta);

/* beattracking */
aubio_beattracking_t * new_aubio_beattracking(uint_t winlen, uint_t channels);
void aubio_beattracking_do(aubio_beattracking_t * bt, fvec_t * dfframes, fvec_t * out);
void del_aubio_beattracking(aubio_beattracking_t * p);
smpl_t aubio_beattracking_get_bpm(aubio_beattracking_t * p);
smpl_t aubio_beattracking_get_confidence(aubio_beattracking_t * p);

/* tempo */
typedef struct _aubio_tempo_t aubio_tempo_t;
aubio_tempo_t * new_aubio_tempo (char_t * mode,
    uint_t buf_size, uint_t hop_size, uint_t channels, uint_t samplerate);
void aubio_tempo_do (aubio_tempo_t *o, fvec_t * input, fvec_t * tempo);
uint_t aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence);
uint_t aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold);
smpl_t aubio_tempo_get_bpm(aubio_tempo_t * bt);
smpl_t aubio_tempo_get_confidence(aubio_tempo_t * bt);
void del_aubio_tempo(aubio_tempo_t * o);

/* sndfile */
%{
#if HAVE_SNDFILE
#include "sndfileio.h"
%}
aubio_sndfile_t * new_aubio_sndfile_ro (const char * inputfile);
aubio_sndfile_t * new_aubio_sndfile_wo(aubio_sndfile_t * existingfile, const char * outputname);
void aubio_sndfile_info(aubio_sndfile_t * file);
int aubio_sndfile_write(aubio_sndfile_t * file, int frames, fvec_t * write);
int aubio_sndfile_read(aubio_sndfile_t * file, int frames, fvec_t * read);
int del_aubio_sndfile(aubio_sndfile_t * file);
uint_t aubio_sndfile_channels(aubio_sndfile_t * file);
uint_t aubio_sndfile_samplerate(aubio_sndfile_t * file);
%{
#endif /* HAVE_SNDFILE */
%}
