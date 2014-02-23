#include "aubiowraphell.h"

typedef struct
{
  PyObject_HEAD
  aubio_sink_t * o;
  char_t* uri;
  uint_t samplerate;
  uint_t channels;
} Py_sink;

static char Py_sink_doc[] = "sink object";

static PyObject *
Py_sink_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
  Py_sink *self;
  char_t* uri = NULL;
  uint_t samplerate = 0;
  uint_t channels = 0;
  static char *kwlist[] = { "uri", "samplerate", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|sII", kwlist,
          &uri, &samplerate, &channels)) {
    return NULL;
  }

  self = (Py_sink *) pytype->tp_alloc (pytype, 0);

  if (self == NULL) {
    return NULL;
  }

  self->uri = "none";
  if (uri != NULL) {
    self->uri = uri;
  }

  self->samplerate = Py_aubio_default_samplerate;
  if ((sint_t)samplerate > 0) {
    self->samplerate = samplerate;
  } else if ((sint_t)samplerate < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative value for samplerate");
    return NULL;
  }

  self->channels = 1;
  if ((sint_t)channels > 0) {
    self->channels = channels;
  } else if ((sint_t)channels < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative or null value for channels");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_sink_init (Py_sink * self, PyObject * args, PyObject * kwds)
{
  if (self->channels == 1) {
    self->o = new_aubio_sink ( self->uri, self->samplerate );
  } else {
    self->o = new_aubio_sink ( self->uri, 0 );
    aubio_sink_preset_channels ( self->o, self->channels );
    aubio_sink_preset_samplerate ( self->o, self->samplerate );
  }
  if (self->o == NULL) {
    PyErr_SetString (PyExc_StandardError, "error creating sink with this uri");
    return -1;
  }
  self->samplerate = aubio_sink_get_samplerate ( self->o );
  self->channels = aubio_sink_get_channels ( self->o );

  return 0;
}

AUBIO_DEL(sink)

/* function Py_sink_do */
static PyObject *
Py_sink_do(Py_sink * self, PyObject * args)
{
  /* input vectors python prototypes */
  PyObject * write_data_obj;

  /* input vectors prototypes */
  fvec_t* write_data;
  uint_t write;


  if (!PyArg_ParseTuple (args, "OI", &write_data_obj, &write)) {
    return NULL;
  }


  /* input vectors parsing */
  write_data = PyAubio_ArrayToCFvec (write_data_obj);

  if (write_data == NULL) {
    return NULL;
  }





  /* compute _do function */
  aubio_sink_do (self->o, write_data, write);

  Py_RETURN_NONE;
}

/* function Py_sink_do */
static PyObject *
Py_sink_do_multi(Py_sink * self, PyObject * args)
{
  /* input vectors python prototypes */
  PyObject * write_data_obj;

  /* input vectors prototypes */
  fmat_t * write_data;
  uint_t write;


  if (!PyArg_ParseTuple (args, "OI", &write_data_obj, &write)) {
    return NULL;
  }


  /* input vectors parsing */
  write_data = PyAubio_ArrayToCFmat (write_data_obj);

  if (write_data == NULL) {
    return NULL;
  }





  /* compute _do function */
  aubio_sink_do_multi (self->o, write_data, write);
  Py_RETURN_NONE;
}

AUBIO_MEMBERS_START(sink)
  {"uri", T_STRING, offsetof (Py_sink, uri), READONLY, ""},
  {"samplerate", T_INT, offsetof (Py_sink, samplerate), READONLY, ""},
  {"channels", T_INT, offsetof (Py_sink, channels), READONLY, ""},
AUBIO_MEMBERS_STOP(sink)

static PyObject *
Pyaubio_sink_close (Py_sink *self, PyObject *unused)
{
  aubio_sink_close (self->o);
  Py_RETURN_NONE;
}

static PyMethodDef Py_sink_methods[] = {
  {"__call__", (PyCFunction) Py_sink_do, METH_VARARGS,
    "x.__call__(vec, write)\n"
    "write monophonic vector to sink"
    ""},
  {"do", (PyCFunction) Py_sink_do, METH_VARARGS,
    "x.do(vec, write)\n"
    "write monophonic vector to sink"
    ""},
  {"do_multi", (PyCFunction) Py_sink_do_multi, METH_VARARGS,
    "x.do_multi(mat, write)\n"
    "write polyphonic vector to sink"},
  {"close", (PyCFunction) Pyaubio_sink_close, METH_NOARGS,
    "x.close()\n"
    "close this sink now"},
  {NULL} /* sentinel */
};

AUBIO_TYPEOBJECT(sink, "aubio.sink")
