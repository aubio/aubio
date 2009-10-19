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
  Py_cvec *vec;
  Py_fvec *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = (Py_fvec*) PyObject_New (Py_fvec, &Py_fvecType);
  output->channels = vec->channels;
  output->length = self->n_filters;
  output->o = new_fvec(self->n_filters, vec->channels);

  // compute the function
  aubio_filterbank_do (self->o, vec->o, output->o);
  return (PyObject *)PyAubio_FvecToArray(output);
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
  Py_fvec *freqs;
  if (!PyArg_ParseTuple (args, "OI", &input, &samplerate)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  freqs = PyAubio_ArrayToFvec (input);

  if (freqs == NULL) {
    return NULL;
  }

  err = aubio_filterbank_set_triangle_bands (self->o,
      freqs->o, samplerate);
  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter to A-weighting");
    return NULL;
  }
  return Py_None;
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
  return Py_None;
}

static PyObject * 
Py_filterbank_get_coeffs (Py_filterbank * self, PyObject *unused)
{
  Py_fvec *output = (Py_fvec *) PyObject_New (Py_fvec, &Py_fvecType);
  output->channels = self->n_filters;
  output->length = self->win_s / 2 + 1;
  output->o = aubio_filterbank_get_coeffs (self->o);
  return (PyObject *)PyAubio_FvecToArray(output);
}

static PyMethodDef Py_filterbank_methods[] = {
  {"set_triangle_bands", (PyCFunction) Py_filterbank_set_triangle_bands,
    METH_VARARGS, "set coefficients of filterbanks"},
  {"set_mel_coeffs_slaney", (PyCFunction) Py_filterbank_set_mel_coeffs_slaney,
    METH_VARARGS, "set coefficients of filterbank as in Auditory Toolbox"},
  {"get_coeffs", (PyCFunction) Py_filterbank_get_coeffs,
    METH_NOARGS, "get coefficients of filterbank"},
  {NULL}
};

AUBIO_TYPEOBJECT(filterbank, "aubio.filterbank")
