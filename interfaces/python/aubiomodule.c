#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include <numpy/arrayobject.h>

#include "aubio-types.h"

static char Py_alpha_norm_doc[] = "compute alpha normalisation factor";

static PyObject *
Py_alpha_norm (PyObject * self, PyObject * args)
{
  PyObject *input;
  Py_fvec *vec;
  smpl_t alpha;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "Of:alpha_norm", &input, &alpha)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue ("f", fvec_alpha_norm (vec->o, alpha));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static char Py_zero_crossing_rate_doc[] = "compute zero crossing rate";

static PyObject *
Py_zero_crossing_rate (PyObject * self, PyObject * args)
{
  PyObject *input;
  Py_fvec *vec;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "O:zero_crossing_rate", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue ("f", aubio_zero_crossing_rate (vec->o));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static char Py_min_removal_doc[] = "compute zero crossing rate";

static PyObject * 
Py_min_removal(PyObject * self, PyObject * args)
{
  PyObject *input;
  Py_fvec *vec;

  if (!PyArg_ParseTuple (args, "O:min_removal", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  fvec_min_removal (vec->o);
  // since this function does not return, we could return None
  //return Py_None;
  // however it is convenient to return the modified vector 
  return (PyObject *) PyAubio_FvecToArray(vec);
  // or even without converting it back to an array
  //Py_INCREF(vec);
  //return (PyObject *)vec;
}

static PyMethodDef aubio_methods[] = {
  {"alpha_norm", Py_alpha_norm, METH_VARARGS, Py_alpha_norm_doc},
  {"zero_crossing_rate", Py_zero_crossing_rate, METH_VARARGS, 
    Py_zero_crossing_rate_doc},
  {"min_removal", Py_min_removal, METH_VARARGS, Py_min_removal_doc},
  {NULL, NULL} /* Sentinel */
};

static char aubio_module_doc[] = "Python module for the aubio library";

PyMODINIT_FUNC
init_aubio (void)
{
  PyObject *m;
  int err;

  if ((PyType_Ready (&Py_fvecType) < 0) ||
      (PyType_Ready (&Py_cvecType) < 0) ||
      (PyType_Ready (&Py_filterType) < 0)) {
    return;
  }

  err = _import_array ();

  if (err != 0) {
    fprintf (stderr,
        "Unable to import Numpy C API from aubio module (error %d)\n", err);
  }

  m = Py_InitModule3 ("_aubio", aubio_methods, aubio_module_doc);

  if (m == NULL) {
    return;
  }

  Py_INCREF (&Py_fvecType);
  PyModule_AddObject (m, "fvec", (PyObject *) & Py_fvecType);
  Py_INCREF (&Py_cvecType);
  PyModule_AddObject (m, "cvec", (PyObject *) & Py_cvecType);
  Py_INCREF (&Py_filterType);
  PyModule_AddObject (m, "digital_filter", (PyObject *) & Py_filterType);
}
