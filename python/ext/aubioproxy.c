#include "aubio-types.h"

fvec_t *
PyAubio_ArrayToCFvec (PyObject *input) {
  PyObject *array;
  fvec_t *vec;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  // parsing input object into a Py_fvec
  if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM ((PyArrayObject *)input) > 1) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than one dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // vec = new_fvec (vec->length);
    // no need to really allocate fvec, just its struct member
    vec = (fvec_t *)malloc(sizeof(fvec_t));
    long length = PyArray_SIZE ((PyArrayObject *)array);
    if (length > 0) {
      vec->length = (uint_t)length;
    } else {
      PyErr_SetString (PyExc_ValueError, "input array size should be greater than 0");
      goto fail;
    }
    vec->data = (smpl_t *) PyArray_GETPTR1 ((PyArrayObject *)array, 0);

  } else if (PyObject_TypeCheck (input, &PyList_Type)) {
    PyErr_SetString (PyExc_ValueError, "does not convert from list yet");
    return NULL;
  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept vector of float as input");
    return NULL;
  }

  return vec;

fail:
  return NULL;
}

PyObject *
PyAubio_CFvecToArray (fvec_t * self)
{
  npy_intp dims[] = { self->length, 1 };
  return PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->data);
}

Py_cvec *
PyAubio_CCvecToPyCvec (cvec_t * input) {
  Py_cvec *vec = (Py_cvec*) PyObject_New (Py_cvec, &Py_cvecType);
  vec->length = input->length;
  vec->o = input;
  Py_INCREF(vec);
  return vec;
}

cvec_t *
PyAubio_ArrayToCCvec (PyObject *input) {
  if (PyObject_TypeCheck (input, &Py_cvecType)) {
      return ((Py_cvec*)input)->o;
  } else {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      return NULL;
  }
}

PyObject *
PyAubio_CFmatToArray (fmat_t * input)
{
  PyObject *array = NULL;
  uint_t i;
  npy_intp dims[] = { input->length, 1 };
  PyObject *concat = PyList_New (0), *tmp = NULL;
  for (i = 0; i < input->height; i++) {
    tmp = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, input->data[i]);
    PyList_Append (concat, tmp);
    Py_DECREF (tmp);
  }
  array = PyArray_FromObject (concat, AUBIO_NPY_SMPL, 2, 2);
  Py_DECREF (concat);
  return array;
}

fmat_t *
PyAubio_ArrayToCFmat (PyObject *input) {
  PyObject *array;
  fmat_t *mat;
  uint_t i;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  // parsing input object into a Py_fvec
  if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM ((PyArrayObject *)input) > 2) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than two dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // no need to really allocate fvec, just its struct member
    mat = (fmat_t *)malloc(sizeof(fmat_t));
    long length = PyArray_DIM ((PyArrayObject *)array, 1);
    if (length > 0) {
      mat->length = (uint_t)length;
    } else {
      PyErr_SetString (PyExc_ValueError, "input array dimension 1 should be greater than 0");
      goto fail;
    }
    long height = PyArray_DIM ((PyArrayObject *)array, 0);
    if (height > 0) {
      mat->height = (uint_t)height;
    } else {
      PyErr_SetString (PyExc_ValueError, "input array dimension 0 should be greater than 0");
      goto fail;
    }
    mat->data = (smpl_t **)malloc(sizeof(smpl_t*) * mat->height);
    for (i=0; i< mat->height; i++) {
      mat->data[i] = (smpl_t*)PyArray_GETPTR1 ((PyArrayObject *)array, i);
    }

  } else if (PyObject_TypeCheck (input, &PyList_Type)) {
    PyErr_SetString (PyExc_ValueError, "can not convert list to fmat");
    return NULL;
  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept matrix of float as input");
    return NULL;
  }

  return mat;

fail:
  return NULL;
}

