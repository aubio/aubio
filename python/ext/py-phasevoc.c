#include "aubio-types.h"

static char Py_pvoc_doc[] = "pvoc object";

typedef struct
{
  PyObject_HEAD
  aubio_pvoc_t * o;
  uint_t win_s;
  uint_t hop_s;
  cvec_t *output;
  fvec_t *routput;
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
    char_t errstr[30];
    sprintf(errstr, "error creating pvoc with %d, %d", self->win_s, self->hop_s);
    PyErr_SetString (PyExc_RuntimeError, errstr);
    return -1;
  }

  self->output = new_cvec(self->win_s);
  self->routput = new_fvec(self->hop_s);

  return 0;
}


static void
Py_pvoc_del (Py_pvoc *self, PyObject *unused)
{
  del_aubio_pvoc(self->o);
  del_cvec(self->output);
  del_fvec(self->routput);
  Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject * 
Py_pvoc_do(Py_pvoc * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  aubio_pvoc_do (self->o, vec, self->output);
  return (PyObject *)PyAubio_CCvecToPyCvec(self->output);
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
  cvec_t *vec;
  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  aubio_pvoc_rdo (self->o, vec, self->routput);
  return (PyObject *)PyAubio_CFvecToArray(self->routput);
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
};
