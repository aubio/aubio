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

PyObject *
Py_aubio_level_lin(PyObject *self, PyObject *args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *level_lin;

  if (!PyArg_ParseTuple (args, "O:level_lin", &input)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);
  if (vec == NULL) {
    return NULL;
  }

  level_lin = Py_BuildValue("f", aubio_level_lin(vec));
  if (level_lin == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing level_lin");
    return NULL;
  }

  return level_lin;
}
