#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include <numpy/arrayobject.h>

#include "aubio-types.h"

Py_fvec *
PyAubio_ArrayToFvec (PyObject *input) {
  PyObject *array;
  Py_fvec *vec;
  uint_t i;
  // parsing input object into a Py_fvec
  if (PyObject_TypeCheck (input, &Py_fvecType)) {
    // input is an fvec, nothing else to do
    vec = (Py_fvec *) input;
  } else if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec 
    if (PyArray_NDIM (input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM (input) > 2) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than two dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT (input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
#if AUBIO_DO_CASTING
    } else if (PyArray_TYPE (input) != AUBIO_FLOAT) {
      // input data type is not float32, casting 
      array = PyArray_Cast ( (PyArrayObject*) input, AUBIO_FLOAT);
      if (array == NULL) {
        PyErr_SetString (PyExc_IndexError, "failed converting to NPY_FLOAT");
        goto fail;
      }
#else
    } else if (PyArray_TYPE (input) != AUBIO_FLOAT) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
#endif
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // create a new fvec object
    vec = (Py_fvec*) PyObject_New (Py_fvec, &Py_fvecType); 
    if (PyArray_NDIM (array) == 1) {
      vec->channels = 1;
      vec->length = PyArray_SIZE (array);
    } else {
      vec->channels = PyArray_DIM (array, 0);
      vec->length = PyArray_DIM (array, 1);
    }

    // no need to really allocate fvec, just its struct member 
    // vec->o = new_fvec (vec->length, vec->channels);
    vec->o = (fvec_t *)malloc(sizeof(fvec_t));
    vec->o->length = vec->length; vec->o->channels = vec->channels;
    vec->o->data = (smpl_t**)malloc(vec->o->channels * sizeof(smpl_t*));
    // hat data[i] point to array line
    for (i = 0; i < vec->channels; i++) {
      vec->o->data[i] = (smpl_t *) PyArray_GETPTR1 (array, i);
    }

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array or fvec as input");
    return NULL;
  }

  return vec;

fail:
  return NULL;
}



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

static PyMethodDef aubio_methods[] = {
  {"alpha_norm", Py_alpha_norm, METH_VARARGS, Py_alpha_norm_doc},
  {NULL, NULL}                  /* Sentinel */
};

static char aubio_module_doc[] = "Python module for the aubio library";

PyMODINIT_FUNC
init_aubio (void)
{
  PyObject *m;
  int err;

  if (PyType_Ready (&Py_fvecType) < 0) {
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
}
