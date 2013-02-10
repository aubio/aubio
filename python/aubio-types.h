#include <Python.h>
#include <structmember.h>
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#define AUBIO_UNSTABLE 1
#include <aubio.h>

#define Py_default_vector_length 1024
#define Py_default_vector_height 1

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

