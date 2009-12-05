#include <Python.h>
#include <structmember.h>
#define NO_IMPORT_ARRAY
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

/**

Defining this constant to 1 will allow PyAubio_CastToFvec to convert from data
types different than NPY_FLOAT to and fvec, and therefore creating a copy of
it. 

*/

typedef struct
{
  PyObject_HEAD
  fvec_t * o;
  uint_t length;
} Py_fvec;
extern PyTypeObject Py_fvecType;
extern PyObject *PyAubio_FvecToArray (Py_fvec * self);
extern PyObject *PyAubio_CFvecToArray (fvec_t * self);
extern Py_fvec *PyAubio_ArrayToFvec (PyObject * self);

typedef struct
{
  PyObject_HEAD
  fmat_t * o;
  uint_t length;
  uint_t height;
} Py_fmat;
extern PyTypeObject Py_fmatType;
extern PyObject *PyAubio_FmatToArray (Py_fmat * self);
extern PyObject *PyAubio_CFmatToArray (fmat_t * self);
extern Py_fmat *PyAubio_ArrayToFmat (PyObject * self);

typedef struct
{
  PyObject_HEAD
  cvec_t * o;
  uint_t length;
  uint_t channels;
} Py_cvec;
extern PyTypeObject Py_cvecType;
extern PyObject *PyAubio_CvecToArray (Py_cvec * self);
extern Py_cvec *PyAubio_ArrayToCvec (PyObject * self);

extern PyTypeObject Py_filterType;

extern PyTypeObject Py_filterbankType;

extern PyTypeObject Py_fftType;

extern PyTypeObject Py_pvocType;
