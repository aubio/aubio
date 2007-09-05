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

#define NYQUIST 22050.f

//libXtract enums
// TODO: remove them 

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
int aubio_mfcc_do(const float *data, const int N, const void *argv, float *result);

/** \brief Extract the Discrete Cosine transform of a time domain signal
 * \param *data: a pointer to the first element in an array of floats representing an audio vector
 * \param N: the number of array elements to be considered
 * \param *argv: a pointer to NULL 
 * \param *result: a pointer to an array containing resultant dct coefficients
 */
int aubio_dct_do(const float *data, const int N, const void *argv, float *result);


#endif