#include "aubio-types.h"

PyObject *
Py_aubio_window(PyObject *self, PyObject *args)
{
  char_t *wintype = NULL;
  uint_t winlen = 0;
  fvec_t *window = NULL;

  if (!PyArg_ParseTuple (args, "|sI", &wintype, &winlen)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  window = new_aubio_window(wintype, winlen);
  if (window == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing window");
    return NULL;
  }

  return (PyObject *) PyAubio_CFvecToArray(window);
}
