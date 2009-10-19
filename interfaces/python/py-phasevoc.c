#include "aubiowraphell.h"

static char Py_pvoc_doc[] = "pvoc object";

AUBIO_DECLARE(pvoc, uint_t win_s; uint_t hop_s; uint_t channels)

//AUBIO_NEW(pvoc)
static PyObject *
Py_pvoc_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0, hop_s = 0, channels = 0;
  Py_pvoc *self;
  static char *kwlist[] = { "win_s", "hop_s", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|III", kwlist,
          &win_s, &hop_s, &channels)) {
    return NULL;
  }

  self = (Py_pvoc *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->win_s = Py_default_vector_length;
  self->hop_s = Py_default_vector_length/2;
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

  if (hop_s > 0) {
    self->hop_s = hop_s;
  } else if (hop_s < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative hop size");
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


AUBIO_INIT(pvoc, self->win_s, self->hop_s, self->channels)

AUBIO_DEL(pvoc)

static PyObject * 
Py_pvoc_do(PyObject * self, PyObject * args)
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
  output->length = ((Py_pvoc *) self)->win_s;
  output->o = new_cvec(((Py_pvoc *) self)->win_s, vec->channels);

  // compute the function
  aubio_pvoc_do (((Py_pvoc *)self)->o, vec->o, output->o);
  Py_INCREF(output);
  return (PyObject *)output;
  //return (PyObject *)PyAubio_CvecToArray(output);
}

AUBIO_MEMBERS_START(pvoc) 
  {"win_s", T_INT, offsetof (Py_pvoc, win_s), READONLY,
    "size of the window"},
  {"hop_s", T_INT, offsetof (Py_pvoc, hop_s), READONLY,
    "size of the hop"},
  {"channels", T_INT, offsetof (Py_pvoc, channels), READONLY,
    "number of channels"},
AUBIO_MEMBERS_STOP(pvoc)

static PyObject * 
Py_pvoc_rdo(PyObject * self, PyObject * args)
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
  output->length = ((Py_pvoc *) self)->hop_s;
  output->o = new_fvec(output->length, output->channels);

  // compute the function
  aubio_pvoc_rdo (((Py_pvoc *)self)->o, vec->o, output->o);
  return (PyObject *)PyAubio_FvecToArray(output);
}

static PyMethodDef Py_pvoc_methods[] = {
  {"rdo", (PyCFunction) Py_pvoc_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

AUBIO_TYPEOBJECT(pvoc, "aubio.pvoc")
