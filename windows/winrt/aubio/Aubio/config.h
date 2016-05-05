/*

config.h for WinRT static library

(does not use libsndfile nor FFTW)

*/

#define HAVE_STDLIB_H 1

#define HAVE_STDIO_H 1

#define HAVE_COMPLEX_H 1

#define _USE_MATH_DEFINES // math inclusion requires a define (https://msdn.microsoft.com/en-us/library/4hwaceh6.aspx)

#define HAVE_MATH_H

#define HAVE_STRING_H 1

#define HAVE_LIMITS_H 1

#define HAVE_C99_VARARGS_MACROS // to make MSVC happy

#define _CRT_SECURE_NO_WARNINGS

#pragma warning( disable : 4244) // conversion from X to Y, possible loss of data

#pragma warning( disable : 4305) // truncation from X to Y
