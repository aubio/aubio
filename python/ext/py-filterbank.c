#include "aubiowraphell.h"

static char Py_filterbank_doc[] = "filterbank object";

AUBIO_DECLARE(filterbank, uint_t n_filters; uint_t win_s)

//AUBIO_NEW(filterbank)
static PyObject *
Py_filterbank_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0, n_filters = 0;
  Py_filterbank *self;
  static char *kwlist[] = { "n_filters", "win_s", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &n_filters, &win_s)) {
    return NULL;
  }

  self = (Py_filterbank *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->win_s = Py_default_vector_length;
  if (win_s > 0) {
    self->win_s = win_s;
  } else if (win_s < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative window size");
    return NULL;
  }

  self->n_filters = 40;
  if (n_filters > 0) {
    self->n_filters = n_filters;
  } else if (n_filters < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of filters");
    return NULL;
  }

  return (PyObject *) self;
}


AUBIO_INIT(filterbank, self->n_filters, self->win_s)

AUBIO_DEL(filterbank)

static PyObject *
Py_filterbank_do(Py_filterbank * self, PyObject * args)
{
  PyObject *input;
  cvec_t *vec;
  fvec_t *out;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  out = new_fvec (self->n_filters);

  // compute the function
  aubio_filterbank_do (self->o, vec, out);
  return (PyObject *)PyAubio_CFvecToArray(out);
}

AUBIO_MEMBERS_START(filterbank)
  {"win_s", T_INT, offsetof (Py_filterbank, win_s), READONLY,
    "size of the window"},
  {"n_filters", T_INT, offsetof (Py_filterbank, n_filters), READONLY,
    "number of filters"},
AUBIO_MEMBERS_STOP(filterbank)

static PyObject *
Py_filterbank_set_triangle_bands (Py_filterbank * self, PyObject *args)
{
  uint_t err = 0;

  PyObject *input;
  uint_t samplerate;
  fvec_t *freqs;
  if (!PyArg_ParseTuple (args, "OI", &input, &samplerate)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  freqs = PyAubio_ArrayToCFvec (input);

  if (freqs == NULL) {
    return NULL;
  }

  err = aubio_filterbank_set_triangle_bands (self->o,
      freqs, samplerate);
  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter to A-weighting");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *
Py_filterbank_set_mel_coeffs_slaney (Py_filterbank * self, PyObject *args)
{
  uint_t err = 0;

  uint_t samplerate;
  if (!PyArg_ParseTuple (args, "I", &samplerate)) {
    return NULL;
  }

  err = aubio_filterbank_set_mel_coeffs_slaney (self->o, samplerate);
  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter to A-weighting");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *
Py_filterbank_set_coeffs (Py_filterbank * self, PyObject *args)
{
  uint_t err = 0;

  PyObject *input;
  fmat_t *coeffs;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  coeffs = PyAubio_ArrayToCFmat (input);

  if (coeffs == NULL) {
    PyErr_SetString (PyExc_ValueError,
        "unable to parse input array");
    return NULL;
  }

  err = aubio_filterbank_set_coeffs (self->o, coeffs);

  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter coefficients");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *
Py_filterbank_get_coeffs (Py_filterbank * self, PyObject *unused)
{
  return (PyObject *)PyAubio_CFmatToArray(
      aubio_filterbank_get_coeffs (self->o) );
}

static PyMethodDef Py_filterbank_methods[] = {
  {"set_triangle_bands", (PyCFunction) Py_filterbank_set_triangle_bands,
    METH_VARARGS, "set coefficients of filterbanks"},
  {"set_mel_coeffs_slaney", (PyCFunction) Py_filterbank_set_mel_coeffs_slaney,
    METH_VARARGS, "set coefficients of filterbank as in Auditory Toolbox"},
  {"get_coeffs", (PyCFunction) Py_filterbank_get_coeffs,
    METH_NOARGS, "get coefficients of filterbank"},
  {"set_coeffs", (PyCFunction) Py_filterbank_set_coeffs,
    METH_VARARGS, "set coefficients of filterbank"},
  {NULL}
};

AUBIO_TYPEOBJECT(filterbank, "aubio.filterbank")
