#include <Python.h>
#include <structmember.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <aubio.h>

#define Py_default_vector_length   1024
#define Py_default_vector_channels 1

#define Py_aubio_default_samplerate 44100

#ifdef HAVE_AUBIO_DOUBLE
#define AUBIO_FLOAT NPY_FLOAT
#else
#define AUBIO_FLOAT NPY_LONG
#endif

/**

Defining this constant to 1 will allow PyAubio_CastToFvec to convert from data
types different than NPY_FLOAT to and fvec, and therefore creating a copy of
it. 

*/
#define AUBIO_DO_CASTING 0

typedef struct
{
  PyObject_HEAD
  fvec_t * o;
  uint_t length;
  uint_t channels;
} Py_fvec;
extern PyTypeObject Py_fvecType;

extern PyTypeObject Py_cvecType;

extern PyTypeObject Py_filterType;

extern PyObject *PyAubio_FvecToArray (Py_fvec * self);

extern Py_fvec *PyAubio_ArrayToFvec (PyObject * self);
