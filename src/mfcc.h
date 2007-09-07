/*
   Copyright (C) 2006 Amaury Hazan
   Ported to aubio from LibXtract
   http://libxtract.sourceforge.net/
   

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

#ifndef MFCC_H 
#define MFCC_H 

#ifdef __cplusplus
extern "C" {
#endif

#include "filterbank.h"

//libXtract constants and enums
// TODO: remove them 

#define XTRACT_SQ(a) ((a) * (a))
#define XTRACT_MIN(a, b) ((a) < (b) ? (a) : (b))
#define XTRACT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define XTRACT_NEEDS_FFTW printf("LibXtract must be compiled with fftw support to use this function.\n")
#define XTRACT_VERY_SMALL_NUMBER 2e-42
#define XTRACT_LOG_LIMIT XTRACT_VERY_SMALL_NUMBER
#define XTRACT_LOG_LIMIT_DB -96.0
#define XTRACT_DB_SCALE_OFFSET 96.0
#define XTRACT_VERY_BIG_NUMBER 2e42
#define XTRACT_SR_UPPER_LIMIT 192000.0
#define XTRACT_SR_LOWER_LIMIT 22050.0
#define XTRACT_SR_DEFAULT 44100.0
#define XTRACT_FUNDAMENTAL_DEFAULT 440.0
#define XTRACT_CHECK_nyquist if(!nyquist) nyquist = XTRACT_SR_DEFAULT / 2
#define XTRACT_CHECK_q if(!q) q = XTRACT_SR_DEFAULT / N
#define XTRACT_IS_ODD(x) (x % 2 != 0 ? 1 : 0) 
#define XTRACT_SR_LIMIT SR_UPPER_LIMIT
#define XTRACT_FFT_BANDS_MIN 16
#define XTRACT_FFT_BANDS_MAX 65536
#define XTRACT_FFT_BANDS_DEF 1024
#define XTRACT_SPEC_BW_MIN 0.168 /* Minimum spectral bandwidth \
            (= SR_LOWER_LIMIT / FFT_BANDS_MAX*/ 
#define XTRACT_SPEC_BW_MAX 12000.0 /* SR_UPPER_LIMIT / FFT_BANDS_MIN */
#define XTRACT_SPEC_BW_DEF 43.066 /* SR_DEFAULT / FFT_BANDS_DEF */

/** \brief Enumeration of feature initialisation functions */
enum xtract_feature_init_ {
    XTRACT_INIT_MFCC = 100,
    XTRACT_INIT_BARK
};

/** \brief Enumeration of feature types */
enum xtract_feature_types_ {
    XTRACT_SCALAR,
    XTRACT_VECTOR,
    XTRACT_DELTA
};

/** \brief Enumeration of mfcc types */
enum xtract_mfcc_types_ {
    XTRACT_EQUAL_GAIN,
    XTRACT_EQUAL_AREA
};

/** \brief Enumeration of return codes */
enum xtract_return_codes_ {
    XTRACT_SUCCESS,
    XTRACT_MALLOC_FAILED,
    XTRACT_BAD_ARGV,
    XTRACT_BAD_VECTOR_SIZE,
    XTRACT_NO_RESULT,
    XTRACT_FEATURE_NOT_IMPLEMENTED
};

/** \brief Enumeration of spectrum types */
enum xtract_spectrum_ {
    XTRACT_MAGNITUDE_SPECTRUM,
    XTRACT_LOG_MAGNITUDE_SPECTRUM,
    XTRACT_POWER_SPECTRUM,
    XTRACT_LOG_POWER_SPECTRUM
};

/** \brief Enumeration of data types*/
typedef enum type_ {
    XTRACT_FLOAT,
    XTRACT_FLOATARRAY,
    XTRACT_INT,
    XTRACT_MEL_FILTER
} xtract_type_t;

/** \brief Enumeration of units*/
typedef enum unit_ {
    /* NONE, ANY */
    XTRACT_HERTZ = 2,
    XTRACT_ANY_AMPLITUDE_HERTZ,
    XTRACT_DBFS,
    XTRACT_DBFS_HERTZ,
    XTRACT_PERCENT,
    XTRACT_SONE
} xtract_unit_t;

/** \brief Boolean */
typedef enum {
    XTRACT_FALSE,
    XTRACT_TRUE
} xtract_bool_t;

/** \brief Enumeration of vector format types*/
typedef enum xtract_vector_ {
    /* N/2 magnitude/log-magnitude/power/log-power coeffs and N/2 frequencies */
    XTRACT_SPECTRAL,     
    /* N spectral amplitudes */
    XTRACT_SPECTRAL_MAGNITUDES, 
    /* N/2 magnitude/log-magnitude/power/log-power peak coeffs and N/2 
     * frequencies */
    XTRACT_SPECTRAL_PEAKS,
    /* N spectral peak amplitudes */
    XTRACT_SPECTRAL_PEAKS_MAGNITUDES,
    /* N spectral peak frequencies */
    XTRACT_SPECTRAL_PEAKS_FREQUENCIES,
    /* N/2 magnitude/log-magnitude/power/log-power harmonic peak coeffs and N/2 
     * frequencies */
    XTRACT_SPECTRAL_HARMONICS,
    /* N spectral harmonic amplitudes */
    XTRACT_SPECTRAL_HARMONICS_MAGNITUDES,
    /* N spectral harmonic frequencies */
    XTRACT_SPECTRAL_HARMONICS_FREQUENCIES,
    XTRACT_ARBITRARY_SERIES,
    XTRACT_AUDIO_SAMPLES,
    XTRACT_MEL_COEFFS, 
    XTRACT_BARK_COEFFS,
    XTRACT_NO_DATA
} xtract_vector_t;




// Computation

/** \brief Extract Mel Frequency Cepstral Coefficients based on a method described by Rabiner
 * 
 * \param *data: a pointer to the first element in an array of spectral magnitudes, e.g. the first half of the array pointed to by *resul from xtract_spectrum()
 * \param N: the number of array elements to be considered
 * \param *argv: a pointer to a data structure of type xtract_mel_filter, containing n_filters coefficient tables to make up a mel-spaced filterbank
 * \param *result: a pointer to an array containing the resultant MFCC
 * 
 * The data structure pointed to by *argv must be obtained by first calling xtract_init_mfcc
 */


int aubio_mfcc_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t *fft_dct, cvec_t *fftgrain_dct);

/** \brief Extract the Discrete Cosine transform of a time domain signal
 * \param *data: a pointer to the first element in an array of floats representing an audio vector
 * \param N: the number of array elements to be considered
 * \param *argv: a pointer to NULL 
 * \param *result: a pointer to an array containing resultant dct coefficients
 */
int aubio_dct_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t *fft_dct, cvec_t *fftgrain_dct);

#ifdef __cplusplus
}
#endif

#endif
