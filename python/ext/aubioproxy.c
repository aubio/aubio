#include "aubio-types.h"

PyObject *
new_py_fvec(uint_t length) {
    npy_intp dims[] = { length, 1 };
    return PyArray_ZEROS(1, dims, AUBIO_NPY_SMPL, 0);
}

PyObject *
new_py_fmat(uint_t height, uint_t length) {
    npy_intp dims[] = { height, length, 1 };
    return PyArray_ZEROS(2, dims, AUBIO_NPY_SMPL, 0);
}

PyObject *
PyAubio_CFvecToArray (fvec_t * self)
{
  npy_intp dims[] = { self->length, 1 };
  return PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->data);
}

/* Checks that input is a non-empty 1-D array of smpl_t. Layout (strides,
 * alignment, byte order, writeability) is checked by the callers below,
 * which differ in what they need. */
int
PyAubio_IsValidVector (PyObject * input) {
  npy_intp length;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    return 0;
  }
  // parsing input object into a Py_fvec
  if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      return 0;
    } else if (PyArray_NDIM ((PyArrayObject *)input) > 1) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than one dimensions");
      return 0;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      return 0;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be " AUBIO_NPY_SMPL_STR);
      return 0;
    }

    length = PyArray_SIZE ((PyArrayObject *)input);
    if (length <= 0) {
      PyErr_SetString (PyExc_ValueError, "input array size should be greater than 0");
      return 0;
    }

  } else if (PyObject_TypeCheck (input, &PyList_Type)) {
    PyErr_SetString (PyExc_ValueError, "does not convert from list yet");
    return 0;
  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept vector of float as input");
    return 0;
  }
  return 1;
}

/* fvec_t and fmat_t point straight at an array's buffer, so the buffer must
 * be laid out as aubio expects: C-contiguous, aligned, native byte order.
 * Strided views such as stereo[:, 0] or x[::2] used to be read as if they
 * were contiguous (aubio#311).
 *
 * Arrays aubio only reads go through the *In functions: an array that is
 * already laid out right is used as is, anything else is copied into one
 * that is. *owner receives a new reference to whichever array is used; the
 * caller releases it (Py_DECREF) once aubio is done with the data.
 *
 * Arrays aubio writes into (outputs, and the in-place functions such as
 * min_removal or shift) go through the plain functions, which never copy:
 * the caller has to see the result in its own array. They must also be
 * writeable (np.frombuffer over bytes is not). */

static PyObject *
PyAubio_InputArray (PyObject *input, int ndim)
{
  PyArray_Descr *descr = PyArray_DescrFromType (AUBIO_NPY_SMPL);
  // steals the reference to descr; returns input itself when no copy is needed
  return PyArray_FromAny (input, descr, ndim, ndim,
      NPY_ARRAY_IN_ARRAY | NPY_ARRAY_NOTSWAPPED, NULL);
}

static int
PyAubio_IsWritableArray (PyArrayObject *array)
{
  if (!PyArray_ISCARRAY (array) || !PyArray_ISNOTSWAPPED (array)) {
    if (!PyArray_ISWRITEABLE (array)) {
      PyErr_SetString (PyExc_ValueError, "array is read-only, but aubio"
          " writes its result into it (pass a writeable copy)");
    } else {
      PyErr_SetString (PyExc_ValueError, "array should be C-contiguous,"
          " aligned and in native byte order, since aubio writes its result"
          " into it (use numpy.ascontiguousarray)");
    }
    return 0;
  }
  return 1;
}

int
PyAubio_ArrayToCFvecIn (PyObject *input, fvec_t *out, PyObject **owner) {
  PyObject *array;
  *owner = NULL;
  if (!PyAubio_IsValidVector(input)) {
    return 0;
  }
  array = PyAubio_InputArray (input, 1);
  if (array == NULL) {
    return 0;
  }
  out->length = (uint_t) PyArray_SIZE ((PyArrayObject *)array);
  out->data = (smpl_t *) PyArray_DATA ((PyArrayObject *)array);
  *owner = array;
  return 1;
}

int
PyAubio_ArrayToCFvec (PyObject *input, fvec_t *out) {

  if (!PyAubio_IsValidVector(input)){
    return 0;
  }
  if (!PyAubio_IsWritableArray ((PyArrayObject *)input)) {
    return 0;
  }

  out->length = (uint_t) PyArray_SIZE ((PyArrayObject *)input);
  out->data = (smpl_t *) PyArray_DATA ((PyArrayObject *)input);
  return 1;
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

static int
PyAubio_IsValidMatrix (PyObject *input) {
  npy_intp length, height;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    return 0;
  }
  // parsing input object into a Py_fvec
  if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      return 0;
    } else if (PyArray_NDIM ((PyArrayObject *)input) != 2) {
      // a 1-D array has no dimension 1 to read the length from
      PyErr_SetString (PyExc_ValueError,
          "input array should have two dimensions");
      return 0;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      return 0;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be " AUBIO_NPY_SMPL_STR);
      return 0;
    }

    length = PyArray_DIM ((PyArrayObject *)input, 1);
    if (length <= 0) {
      PyErr_SetString (PyExc_ValueError, "input array dimension 1 should be greater than 0");
      return 0;
    }
    height = PyArray_DIM ((PyArrayObject *)input, 0);
    if (height <= 0) {
      PyErr_SetString (PyExc_ValueError, "input array dimension 0 should be greater than 0");
      return 0;
    }

  } else if (PyObject_TypeCheck (input, &PyList_Type)) {
    PyErr_SetString (PyExc_ValueError, "can not convert list to fmat");
    return 0;
  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept matrix of float as input");
    return 0;
  }
  return 1;
}

// point mat's rows into a C-contiguous 2-D array
static int
PyAubio_SetCFmatRows (PyArrayObject *array, fmat_t *mat) {
  uint_t i, new_height = (uint_t)PyArray_DIM (array, 0);
  if (mat->height != new_height) {
    if (mat->data) {
      free(mat->data);
    }
    mat->data = (smpl_t **)malloc(sizeof(smpl_t*) * new_height);
    if (!mat->data) {
      mat->height = 0;
      PyErr_NoMemory ();
      return 0;
    }
  }
  mat->height = new_height;
  mat->length = (uint_t)PyArray_DIM (array, 1);
  for (i=0; i< mat->height; i++) {
    mat->data[i] = (smpl_t*)PyArray_GETPTR1 (array, i);
  }
  return 1;
}

int
PyAubio_ArrayToCFmatIn (PyObject *input, fmat_t *mat, PyObject **owner) {
  PyObject *array;
  *owner = NULL;
  if (!PyAubio_IsValidMatrix (input)) {
    return 0;
  }
  array = PyAubio_InputArray (input, 2);
  if (array == NULL) {
    return 0;
  }
  if (!PyAubio_SetCFmatRows ((PyArrayObject *)array, mat)) {
    Py_DECREF (array);
    return 0;
  }
  *owner = array;
  return 1;
}

int
PyAubio_ArrayToCFmat (PyObject *input, fmat_t *mat) {
  if (!PyAubio_IsValidMatrix (input)) {
    return 0;
  }
  if (!PyAubio_IsWritableArray ((PyArrayObject *)input)) {
    return 0;
  }
  return PyAubio_SetCFmatRows ((PyArrayObject *)input, mat);
}
