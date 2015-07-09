#include "aubio-types.h"

PyObject *
Py_aubio_window(PyObject *self, PyObject *args)
{
  PyObject *output = NULL;
  char_t *wintype = NULL;
  uint_t winlen = 0;
  fvec_t *window;

  if (!PyArg_ParseTuple (args, "|sd", &wintype, &winlen)) {
    PyErr_SetString (PyExc_ValueError,
        "failed parsing arguments");
    return NULL;
  }

  //return (PyObject *) PyAubio_CFvecToArray(vec);
  Py_RETURN_NONE;
}
