#include "aubio-types.h"

static char Py_fft_doc[] = "fft object";

typedef struct
{
  PyObject_HEAD
  aubio_fft_t * o;
  uint_t win_s;
  cvec_t *out;
  fvec_t *rout;
} Py_fft;

static PyObject *
Py_fft_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0;
  Py_fft *self;
  static char *kwlist[] = { "win_s", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|I", kwlist,
          &win_s)) {
    return NULL;
  }

  self = (Py_fft *) type->tp_alloc (type, 0);

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

  return (PyObject *) self;
}

static int
Py_fft_init (Py_fft * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_fft (self->win_s);
  if (self->o == NULL) {
    char_t errstr[30];
    sprintf(errstr, "error creating fft with win_s=%d", self->win_s);
    PyErr_SetString (PyExc_Exception, errstr);
    return -1;
  }
  self->out = new_cvec(self->win_s);
  self->rout = new_fvec(self->win_s);

  return 0;
}

static void
Py_fft_del (Py_fft *self, PyObject *unused)
{
  del_aubio_fft(self->o);
  del_cvec(self->out);
  del_fvec(self->rout);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * 
Py_fft_do(Py_fft * self, PyObject * args)
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
  aubio_fft_do (((Py_fft *)self)->o, vec, self->out);
  return (PyObject *)PyAubio_CCvecToPyCvec(self->out);
}

static PyMemberDef Py_fft_members[] = {
  {"win_s", T_INT, offsetof (Py_fft, win_s), READONLY,
    "size of the window"},
  {NULL}
};

static PyObject * 
Py_fft_rdo(Py_fft * self, PyObject * args)
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
  aubio_fft_rdo (((Py_fft *)self)->o, vec, self->rout);
  return (PyObject *)PyAubio_CFvecToArray(self->rout);
}

static PyMethodDef Py_fft_methods[] = {
  {"rdo", (PyCFunction) Py_fft_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

PyTypeObject Py_fftType = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "aubio.fft",
  sizeof (Py_fft),
  0,
  (destructor) Py_fft_del,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  (ternaryfunc)Py_fft_do,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  Py_fft_doc,
  0,
  0,
  0,
  0,
  0,
  0,
  Py_fft_methods,
  Py_fft_members,
  0,
  0,
  0,
  0,
  0,
  0,
  (initproc) Py_fft_init,
  0,
  Py_fft_new,
};
