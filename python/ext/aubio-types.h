#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// define numpy unique symbols for aubio
#define PY_ARRAY_UNIQUE_SYMBOL PYAUBIO_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL PYAUBIO_UFUNC_API

// only import array and ufunc from main module
#ifndef PY_AUBIO_MODULE_MAIN
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>
#ifndef PY_AUBIO_MODULE_UFUNC
#define NO_IMPORT_UFUNC
#else
#include <numpy/ufuncobject.h>
#endif

//#include <numpy/npy_3kcompat.h>

// import aubio
#define AUBIO_UNSTABLE 1
#ifdef USE_LOCAL_AUBIO
#include "aubio.h"
#else
#include "aubio/aubio.h"
#endif

#define Py_default_vector_length 1024

#define Py_aubio_default_samplerate 44100

#if HAVE_AUBIO_DOUBLE
#error "Ouch! Python interface for aubio has not been much tested yet."
#define AUBIO_NPY_SMPL NPY_DOUBLE
#else
#define AUBIO_NPY_SMPL NPY_FLOAT
#endif

// special python type for cvec
typedef struct
{
  PyObject_HEAD
  cvec_t * o;
  uint_t length;
  uint_t channels;
} Py_cvec;
extern PyTypeObject Py_cvecType;

// defined in aubio-proxy.c
extern PyObject *PyAubio_CFvecToArray (fvec_t * self);
extern fvec_t *PyAubio_ArrayToCFvec (PyObject * self);

extern Py_cvec *PyAubio_CCvecToPyCvec (cvec_t * self);
extern cvec_t *PyAubio_ArrayToCCvec (PyObject *input);

extern PyObject *PyAubio_CFmatToArray (fmat_t * self);
extern fmat_t *PyAubio_ArrayToCFmat (PyObject *input);

// hand written wrappers
extern PyTypeObject Py_filterType;

extern PyTypeObject Py_filterbankType;

extern PyTypeObject Py_fftType;

extern PyTypeObject Py_pvocType;

extern PyTypeObject Py_sourceType;

extern PyTypeObject Py_sinkType;
