#include <Python.h>
#include <structmember.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <aubio.h>

typedef struct
{
  PyObject_HEAD fvec_t * o;
  uint_t length;
  uint_t channels;
} Py_fvec;
extern PyTypeObject Py_fvecType;
