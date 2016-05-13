#include "aubio-types.h"

static char Py_pvoc_doc[] = "pvoc object";

typedef struct
{
  PyObject_HEAD
  aubio_pvoc_t * o;
  uint_t win_s;
  uint_t hop_s;
  fvec_t vecin;
  cvec_t cvecin;
  PyObject *output;
  cvec_t c_output;
  PyObject *routput;
  fvec_t c_routput;
} Py_pvoc;


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

static int
Py_pvoc_init (Py_pvoc * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_pvoc ( self->win_s, self->hop_s);
  if (self->o == NULL) {
    PyErr_Format(PyExc_RuntimeError,
        "failed creating pvoc with win_s=%d, hop_s=%d",
        self->win_s, self->hop_s);
    return -1;
  }

  self->output = new_py_cvec(self->win_s);
  self->routput = new_py_fvec(self->hop_s);

  return 0;
}


static void
Py_pvoc_del (Py_pvoc *self, PyObject *unused)
{
  Py_XDECREF(self->output);
  Py_XDECREF(self->routput);
  if (self->o) {
    del_aubio_pvoc(self->o);
  }
  Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
Py_pvoc_do(Py_pvoc * self, PyObject * args)
{
  PyObject *input;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec (input, &(self->vecin) )) {
    return NULL;
  }

  if (self->vecin.length != self->hop_s) {
    PyErr_Format(PyExc_ValueError,
                 "input fvec has length %d, but pvoc expects length %d",
                 self->vecin.length, self->hop_s);
    return NULL;
  }

  Py_INCREF(self->output);
  if (!PyAubio_PyCvecToCCvec (self->output, &(self->c_output))) {
    return NULL;
  }
  // compute the function
  aubio_pvoc_do (self->o, &(self->vecin), &(self->c_output));
  return self->output;
}

static PyMemberDef Py_pvoc_members[] = {
  {"win_s", T_INT, offsetof (Py_pvoc, win_s), READONLY,
    "size of the window"},
  {"hop_s", T_INT, offsetof (Py_pvoc, hop_s), READONLY,
    "size of the hop"},
  { NULL } // sentinel
};

static PyObject *
Py_pvoc_rdo(Py_pvoc * self, PyObject * args)
{
  PyObject *input;
  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_PyCvecToCCvec (input, &(self->cvecin) )) {
    return NULL;
  }

  if (self->cvecin.length != self->win_s / 2 + 1) {
    PyErr_Format(PyExc_ValueError,
                 "input cvec has length %d, but pvoc expects length %d",
                 self->cvecin.length, self->win_s / 2 + 1);
    return NULL;
  }

  Py_INCREF(self->routput);
  if (!PyAubio_ArrayToCFvec(self->routput, &(self->c_routput)) ) {
    return NULL;
  }
  // compute the function
  aubio_pvoc_rdo (self->o, &(self->cvecin), &(self->c_routput));
  return self->routput;
}

static PyMethodDef Py_pvoc_methods[] = {
  {"rdo", (PyCFunction) Py_pvoc_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

PyTypeObject Py_pvocType = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "aubio.pvoc",
  sizeof (Py_pvoc),
  0,
  (destructor) Py_pvoc_del,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  (ternaryfunc)Py_pvoc_do,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  Py_pvoc_doc,
  0,
  0,
  0,
  0,
  0,
  0,
  Py_pvoc_methods,
  Py_pvoc_members,
  0,
  0,
  0,
  0,
  0,
  0,
  (initproc) Py_pvoc_init,
  0,
  Py_pvoc_new,
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
