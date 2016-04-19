#include "aubio-types.h"

static char Py_filterbank_doc[] = "filterbank object";

typedef struct
{
  PyObject_HEAD
  aubio_filterbank_t * o;
  uint_t n_filters;
  uint_t win_s;
  fvec_t *out;
} Py_filterbank;

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

static int
Py_filterbank_init (Py_filterbank * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_filterbank (self->n_filters, self->win_s);
  if (self->o == NULL) {
    char_t errstr[30];
    sprintf(errstr, "error creating filterbank with n_filters=%d, win_s=%d",
        self->n_filters, self->win_s);
    PyErr_SetString (PyExc_RuntimeError, errstr);
    return -1;
  }
  self->out = new_fvec(self->n_filters);

  return 0;
}

static void
Py_filterbank_del (Py_filterbank *self, PyObject *unused)
{
  del_aubio_filterbank(self->o);
  del_fvec(self->out);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Py_filterbank_do(Py_filterbank * self, PyObject * args)
{
  PyObject *input;
  cvec_t *vec;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  aubio_filterbank_do (self->o, vec, self->out);
  return (PyObject *)PyAubio_CFvecToArray(self->out);
}

static PyMemberDef Py_filterbank_members[] = {
  {"win_s", T_INT, offsetof (Py_filterbank, win_s), READONLY,
    "size of the window"},
  {"n_filters", T_INT, offsetof (Py_filterbank, n_filters), READONLY,
    "number of filters"},
  {NULL} /* sentinel */
};

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

PyTypeObject Py_filterbankType = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "aubio.filterbank",
  sizeof (Py_filterbank),
  0,
  (destructor) Py_filterbank_del,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  (ternaryfunc)Py_filterbank_do,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  Py_filterbank_doc,
  0,
  0,
  0,
  0,
  0,
  0,
  Py_filterbank_methods,
  Py_filterbank_members,
  0,
  0,
  0,
  0,
  0,
  0,
  (initproc) Py_filterbank_init,
  0,
  Py_filterbank_new,
};
