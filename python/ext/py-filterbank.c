#include "aubio-types.h"

static char Py_filterbank_doc[] = "filterbank object";

typedef struct
{
  PyObject_HEAD
  aubio_filterbank_t * o;
  uint_t n_filters;
  uint_t win_s;
  cvec_t vec;
  fvec_t freqs;
  fmat_t coeffs;
  PyObject *out;
  fvec_t c_out;
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
    PyErr_Format(PyExc_RuntimeError, "error creating filterbank with"
        " n_filters=%d, win_s=%d", self->n_filters, self->win_s);
    return -1;
  }
  self->out = new_py_fvec(self->n_filters);

  return 0;
}

static void
Py_filterbank_del (Py_filterbank *self, PyObject *unused)
{
  if (self->o) {
    free(self->coeffs.data);
    del_aubio_filterbank(self->o);
  }
  Py_XDECREF(self->out);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Py_filterbank_do(Py_filterbank * self, PyObject * args)
{
  PyObject *input;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_PyCvecToCCvec(input, &(self->vec) )) {
    return NULL;
  }

  if (self->vec.length != self->win_s / 2 + 1) {
    PyErr_Format(PyExc_ValueError,
                 "input cvec has length %d, but fft expects length %d",
                 self->vec.length, self->win_s / 2 + 1);
    return NULL;
  }

  Py_INCREF(self->out);
  if (!PyAubio_ArrayToCFvec(self->out, &(self->c_out))) {
    return NULL;
  }
  // compute the function
  aubio_filterbank_do (self->o, &(self->vec), &(self->c_out));
  return self->out;
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
  if (!PyArg_ParseTuple (args, "OI", &input, &samplerate)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec(input, &(self->freqs) )) {
    return NULL;
  }

  err = aubio_filterbank_set_triangle_bands (self->o,
      &(self->freqs), samplerate);
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
  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFmat(input, &(self->coeffs))) {
    return NULL;
  }

  err = aubio_filterbank_set_coeffs (self->o, &(self->coeffs));

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
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};
