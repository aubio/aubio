#include "aubiowraphell.h"

static char Py_fft_doc[] = "fft object";

AUBIO_DECLARE(fft, uint_t win_s; uint_t channels)

//AUBIO_NEW(fft)
static PyObject *
Py_fft_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0, channels = 0;
  Py_fft *self;
  static char *kwlist[] = { "win_s", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &win_s, &channels)) {
    return NULL;
  }

  self = (Py_fft *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->win_s = Py_default_vector_length;
  self->channels = Py_default_vector_channels;

  if (self == NULL) {
    return NULL;
  }

  if (win_s > 0) {
    self->win_s = win_s;
  } else if (win_s < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative window size");
    return NULL;
  }

  if (channels > 0) {
    self->channels = channels;
  } else if (channels < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of filters");
    return NULL;
  }

  return (PyObject *) self;
}


AUBIO_INIT(fft, self->win_s, self->channels)

AUBIO_DEL(fft)

static PyObject * 
Py_fft_do(PyObject * self, PyObject * args)
{
  PyObject *input;
  Py_fvec *vec;
  Py_cvec *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = (Py_cvec*) PyObject_New (Py_cvec, &Py_cvecType);
  output->channels = vec->channels;
  output->length = ((Py_fft *) self)->win_s;
  output->o = new_cvec(((Py_fft *) self)->win_s, vec->channels);

  // compute the function
  aubio_fft_do (((Py_fft *)self)->o, vec->o, output->o);
  Py_INCREF(output);
  return (PyObject *)output;
  //return (PyObject *)PyAubio_CvecToArray(output);
}

AUBIO_MEMBERS_START(fft) 
  {"win_s", T_INT, offsetof (Py_fft, win_s), READONLY,
    "size of the window"},
  {"channels", T_INT, offsetof (Py_fft, channels), READONLY,
    "number of channels"},
AUBIO_MEMBERS_STOP(fft)

static PyObject * 
Py_fft_rdo(PyObject * self, PyObject * args)
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
  output->length = ((Py_fft *) self)->win_s;
  output->o = new_fvec(output->length, output->channels);

  // compute the function
  aubio_fft_rdo (((Py_fft *)self)->o, vec->o, output->o);
  return (PyObject *)PyAubio_FvecToArray(output);
}

static PyMethodDef Py_fft_methods[] = {
  {"rdo", (PyCFunction) Py_fft_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

AUBIO_TYPEOBJECT(fft, "aubio.fft")
