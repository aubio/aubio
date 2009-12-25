#include "aubiowraphell.h"

static char Py_pvoc_doc[] = "pvoc object";

AUBIO_DECLARE(pvoc, uint_t win_s; uint_t hop_s)

//AUBIO_NEW(pvoc)
static PyObject *
Py_pvoc_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0, hop_s = 0;
  Py_pvoc *self;
  static char *kwlist[] = { "win_s", "hop_s", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &win_s, &hop_s)) {
    return NULL;
  }

  self = (Py_pvoc *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->win_s = Py_default_vector_length;
  self->hop_s = Py_default_vector_length/2;

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

  return (PyObject *) self;
}


AUBIO_INIT(pvoc, self->win_s, self->hop_s)

AUBIO_DEL(pvoc)

static PyObject * 
Py_pvoc_do(Py_pvoc * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;
  cvec_t *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = new_cvec(self->win_s);

  // compute the function
  aubio_pvoc_do (self->o, vec, output);
  return (PyObject *)PyAubio_CCvecToPyCvec(output);
}

AUBIO_MEMBERS_START(pvoc) 
  {"win_s", T_INT, offsetof (Py_pvoc, win_s), READONLY,
    "size of the window"},
  {"hop_s", T_INT, offsetof (Py_pvoc, hop_s), READONLY,
    "size of the hop"},
AUBIO_MEMBERS_STOP(pvoc)

static PyObject * 
Py_pvoc_rdo(Py_pvoc * self, PyObject * args)
{
  PyObject *input;
  cvec_t *vec;
  fvec_t *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = new_fvec(self->hop_s);

  // compute the function
  aubio_pvoc_rdo (self->o, vec, output);
  return (PyObject *)PyAubio_CFvecToArray(output);
}

static PyMethodDef Py_pvoc_methods[] = {
  {"rdo", (PyCFunction) Py_pvoc_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

AUBIO_TYPEOBJECT(pvoc, "aubio.pvoc")
