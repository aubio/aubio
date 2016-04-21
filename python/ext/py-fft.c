#include "aubio-types.h"

static char Py_fft_doc[] = "fft object";

typedef struct
{
  PyObject_HEAD
  aubio_fft_t * o;
  uint_t win_s;
  fvec_t *vecin;
  cvec_t *out;
  Py_cvec *py_out;
  cvec_t *cvecin;
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

  self->cvecin = (cvec_t *)malloc(sizeof(cvec_t));
  self->vecin = (fvec_t *)malloc(sizeof(fvec_t));

  self->out = new_cvec(self->win_s);
  self->py_out = (Py_cvec*) PyObject_New (Py_cvec, &Py_cvecType);
  Py_XINCREF(self->py_out);
  self->rout = new_fvec(self->win_s);

  return 0;
}

static void
Py_fft_del (Py_fft *self, PyObject *unused)
{
  Py_XDECREF((PyObject*)(self->py_out));
  del_aubio_fft(self->o);
  del_cvec(self->out);
  del_fvec(self->rout);
  free(self->cvecin);
  free(self->vecin);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Py_fft_do(Py_fft * self, PyObject * args)
{
  PyObject *input;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec(input, self->vecin)) {
    return NULL;
  }

  // compute the function
  aubio_fft_do (((Py_fft *)self)->o, self->vecin, self->out);
#if 0
  Py_cvec * py_out = (Py_cvec*) PyObject_New (Py_cvec, &Py_cvecType);
  PyObject* output = PyAubio_CCvecToPyCvec(self->out, py_out);
  return output;
#else
  // convert cvec to py_cvec, incrementing refcount to keep a copy
  return PyAubio_CCvecToPyCvec(self->out, self->py_out);
#endif
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

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  if (!PyAubio_ArrayToCCvec (input, self->cvecin) ) {
    return NULL;
  }

  // compute the function
  aubio_fft_rdo (self->o, self->cvecin, self->rout);
  return PyAubio_CFvecToArray(self->rout);
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
