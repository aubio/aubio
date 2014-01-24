#include "aubiowraphell.h"

typedef struct
{
  PyObject_HEAD
  aubio_sink_t * o;
  char_t* uri;
  uint_t samplerate;
} Py_sink;

static char Py_sink_doc[] = "sink object";

static PyObject *
Py_sink_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
  Py_sink *self;
  char_t* uri = NULL;
  uint_t samplerate = 0;
  static char *kwlist[] = { "uri", "samplerate", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|sI", kwlist,
          &uri, &samplerate)) {
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

  return (PyObject *) self;
}

AUBIO_INIT(sink , self->uri, self->samplerate)

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

AUBIO_MEMBERS_START(sink)
  {"uri", T_STRING, offsetof (Py_sink, uri), READONLY, ""},
  {"samplerate", T_INT, offsetof (Py_sink, samplerate), READONLY, ""},
AUBIO_MEMBERS_STOP(sink)


static PyMethodDef Py_sink_methods[] = {
  {NULL} /* sentinel */
};

AUBIO_TYPEOBJECT(sink, "aubio.sink")
