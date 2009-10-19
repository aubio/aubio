#include "aubio-types.h"

typedef struct
{
  PyObject_HEAD
  aubio_filter_t * o;
  uint_t order;
  uint_t channels;
} Py_filter;

static char Py_filter_doc[] = "filter object";

static PyObject *
Py_filter_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int order= 0, channels = 0;
  Py_filter *self;
  static char *kwlist[] = { "order", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &order, &channels)) {
    return NULL;
  }

  self = (Py_filter *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->order = 7;
  self->channels = Py_default_vector_channels;

  if (order > 0) {
    self->order = order;
  } else if (order < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative order");
    return NULL;
  }

  if (channels > 0) {
    self->channels = channels;
  } else if (channels < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of channels");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_filter_init (Py_filter * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_filter (self->order, self->channels);
  if (self->o == NULL) {
    return -1;
  }

  return 0;
}

static void
Py_filter_del (Py_filter * self)
{
  del_aubio_filter (self->o);
  self->ob_type->tp_free ((PyObject *) self);
}

static PyObject * 
Py_filter_do(PyObject * self, PyObject * args)
{
  PyObject *input;
  Py_fvec *vec;

  if (!PyArg_ParseTuple (args, "O:digital_filter.do", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
#if 1
  aubio_filter_do (((Py_filter *)self)->o, vec->o);
  Py_INCREF(vec);
  return (PyObject *)vec;
#else
  Py_fvec *copy = (Py_fvec*) PyObject_New (Py_fvec, &Py_fvecType);
  copy->o = new_fvec(vec->o->length, vec->o->channels);
  aubio_filter_do_outplace (((Py_filter *)self)->o, vec->o, copy->o);
  return (PyObject *)copy;
#endif
}

static PyObject * 
Py_filter_set_c_weighting (Py_filter * self, PyObject *args)
{
  uint_t err = 0;
  uint_t samplerate;
  if (!PyArg_ParseTuple (args, "I", &samplerate)) {
    return NULL;
  }

  err = aubio_filter_set_c_weighting (self->o, samplerate);
  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter to A-weighting");
    return NULL;
  }
  return Py_None;
}

static PyObject * 
Py_filter_set_a_weighting (Py_filter * self, PyObject *args)
{
  uint_t err = 0;
  uint_t samplerate;
  if (!PyArg_ParseTuple (args, "I", &samplerate)) {
    return NULL;
  }

  err = aubio_filter_set_a_weighting (self->o, samplerate);
  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when setting filter to A-weighting");
    return NULL;
  }
  return Py_None;
}

static PyMemberDef Py_filter_members[] = {
  // TODO remove READONLY flag and define getter/setter
  {"order", T_INT, offsetof (Py_filter, order), READONLY,
      "order of the filter"},
  {"channels", T_INT, offsetof (Py_filter, channels), READONLY,
      "number of channels"},
  {NULL}                        /* Sentinel */
};

static PyMethodDef Py_filter_methods[] = {
  {"set_c_weighting", (PyCFunction) Py_filter_set_c_weighting, METH_NOARGS,
      "set filter coefficients to C-weighting"},
  {"set_a_weighting", (PyCFunction) Py_filter_set_a_weighting, METH_NOARGS,
      "set filter coefficients to A-weighting"},
  {NULL}
};

PyTypeObject Py_filterType = {
  PyObject_HEAD_INIT (NULL)
  0,                            /* ob_size           */
  "aubio.digital_filter",       /* tp_name           */
  sizeof (Py_filter),           /* tp_basicsize      */
  0,                            /* tp_itemsize       */
  (destructor) Py_filter_del,   /* tp_dealloc        */
  0,                            /* tp_print          */
  0,                            /* tp_getattr        */
  0,                            /* tp_setattr        */
  0,                            /* tp_compare        */
  0, //(reprfunc) Py_filter_repr,    /* tp_repr           */
  0,                            /* tp_as_number      */
  0,                            /* tp_as_sequence    */
  0,                            /* tp_as_mapping     */
  0,                            /* tp_hash           */
  (ternaryfunc)Py_filter_do,    /* tp_call           */
  0,                            /* tp_str            */
  0,                            /* tp_getattro       */
  0,                            /* tp_setattro       */
  0,                            /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT,           /* tp_flags          */
  Py_filter_doc,                /* tp_doc            */
  0,                            /* tp_traverse       */
  0,                            /* tp_clear          */
  0,                            /* tp_richcompare    */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter           */
  0,                            /* tp_iternext       */
  Py_filter_methods,            /* tp_methods        */
  Py_filter_members,            /* tp_members        */
  0,                            /* tp_getset         */
  0,                            /* tp_base           */
  0,                            /* tp_dict           */
  0,                            /* tp_descr_get      */
  0,                            /* tp_descr_set      */
  0,                            /* tp_dictoffset     */
  (initproc) Py_filter_init,    /* tp_init           */
  0,                            /* tp_alloc          */
  Py_filter_new,                /* tp_new            */
};
