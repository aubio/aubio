#include "aubiowraphell.h"

typedef struct
{
  PyObject_HEAD
  aubio_source_t * o;
  char_t* uri;
  uint_t samplerate;
  uint_t channels;
  uint_t hop_size;
} Py_source;

static char Py_source_doc[] = ""
"   __new__(path, samplerate = 0, hop_size = 512, channels = 1)\n"
"\n"
"       Create a new source, opening the given path for reading.\n"
"\n"
"       Examples\n"
"       --------\n"
"\n"
"       Create a new source, using the original samplerate, with hop_size = 512:\n"
"\n"
"       >>> source('/tmp/t.wav')\n"
"\n"
"       Create a new source, resampling the original to 8000Hz:\n"
"\n"
"       >>> source('/tmp/t.wav', samplerate = 8000)\n"
"\n"
"       Create a new source, resampling it at 32000Hz, hop_size = 32:\n"
"\n"
"       >>> source('/tmp/t.wav', samplerate = 32000, hop_size = 32)\n"
"\n"
"       Create a new source, using its original samplerate:\n"
"\n"
"       >>> source('/tmp/t.wav', samplerate = 0)\n"
"\n"
"   __call__()\n"
"       vec, read = x() <==> vec, read = x.do()\n"
"\n"
"       Read vector from source.\n"
"\n"
"       See also\n"
"       --------\n"
"       aubio.source.do\n"
"\n";

static char Py_source_get_samplerate_doc[] = ""
"x.get_samplerate() -> source samplerate\n"
"\n"
"Get samplerate of source.";

static char Py_source_get_channels_doc[] = ""
"x.get_channels() -> number of channels\n"
"\n"
"Get number of channels in source.";

static char Py_source_do_doc[] = ""
"vec, read = x.do() <==> vec, read = x()\n"
"\n"
"Read monophonic vector from source.";

static char Py_source_do_multi_doc[] = ""
"mat, read = x.do_multi()\n"
"\n"
"Read polyphonic vector from source.";

static char Py_source_close_doc[] = ""
"x.close()\n"
"\n"
"Close this source now.";

static PyObject *
Py_source_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
  Py_source *self;
  char_t* uri = NULL;
  uint_t samplerate = 0;
  uint_t hop_size = 0;
  static char *kwlist[] = { "uri", "samplerate", "hop_size", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|sII", kwlist,
          &uri, &samplerate, &hop_size)) {
    return NULL;
  }

  self = (Py_source *) pytype->tp_alloc (pytype, 0);

  if (self == NULL) {
    return NULL;
  }

  self->uri = "none";
  if (uri != NULL) {
    self->uri = uri;
  }

  self->samplerate = 0;
  if ((sint_t)samplerate > 0) {
    self->samplerate = samplerate;
  } else if ((sint_t)samplerate < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative value for samplerate");
    return NULL;
  }

  self->hop_size = Py_default_vector_length / 2;
  if ((sint_t)hop_size > 0) {
    self->hop_size = hop_size;
  } else if ((sint_t)hop_size < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative value for hop_size");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_source_init (Py_source * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_source ( self->uri, self->samplerate, self->hop_size );
  if (self->o == NULL) {
    PyErr_SetString (PyExc_StandardError, "error creating object");
    return -1;
  }
  self->samplerate = aubio_source_get_samplerate ( self->o );
  self->channels = aubio_source_get_channels ( self->o );

  return 0;
}

AUBIO_DEL(source)

/* function Py_source_do */
static PyObject *
Py_source_do(Py_source * self, PyObject * args)
{


  /* output vectors prototypes */
  fvec_t* read_to;
  uint_t read;






  /* creating output read_to as a new_fvec of length self->hop_size */
  read_to = new_fvec (self->hop_size);
  read = 0;


  /* compute _do function */
  aubio_source_do (self->o, read_to, &read);

  PyObject *outputs = PyList_New(0);
  PyList_Append( outputs, (PyObject *)PyAubio_CFvecToArray (read_to));
  //del_fvec (read_to);
  PyList_Append( outputs, (PyObject *)PyInt_FromLong (read));
  return outputs;
}

/* function Py_source_do_multi */
static PyObject *
Py_source_do_multi(Py_source * self, PyObject * args)
{


  /* output vectors prototypes */
  fmat_t* read_to;
  uint_t read;






  /* creating output read_to as a new_fvec of length self->hop_size */
  read_to = new_fmat (self->channels, self->hop_size);
  read = 0;


  /* compute _do function */
  aubio_source_do_multi (self->o, read_to, &read);

  PyObject *outputs = PyList_New(0);
  PyList_Append( outputs, (PyObject *)PyAubio_CFmatToArray (read_to));
  //del_fvec (read_to);
  PyList_Append( outputs, (PyObject *)PyInt_FromLong (read));
  return outputs;
}

AUBIO_MEMBERS_START(source)
  {"uri", T_STRING, offsetof (Py_source, uri), READONLY,
    "path at which the source was created"},
  {"samplerate", T_INT, offsetof (Py_source, samplerate), READONLY,
    "samplerate at which the source is viewed"},
  {"channels", T_INT, offsetof (Py_source, channels), READONLY,
    "number of channels found in the source"},
  {"hop_size", T_INT, offsetof (Py_source, hop_size), READONLY,
    "number of consecutive frames that will be read at each do or do_multi call"},
AUBIO_MEMBERS_STOP(source)


static PyObject *
Pyaubio_source_get_samplerate (Py_source *self, PyObject *unused)
{
  uint_t tmp = aubio_source_get_samplerate (self->o);
  return (PyObject *)PyInt_FromLong (tmp);
}

static PyObject *
Pyaubio_source_get_channels (Py_source *self, PyObject *unused)
{
  uint_t tmp = aubio_source_get_channels (self->o);
  return (PyObject *)PyInt_FromLong (tmp);
}

static PyObject *
Pyaubio_source_close (Py_source *self, PyObject *unused)
{
  aubio_source_close (self->o);
  Py_RETURN_NONE;
}

static PyMethodDef Py_source_methods[] = {
  {"get_samplerate", (PyCFunction) Pyaubio_source_get_samplerate,
    METH_NOARGS, Py_source_get_samplerate_doc},
  {"get_channels", (PyCFunction) Pyaubio_source_get_channels,
    METH_NOARGS, Py_source_get_channels_doc},
  {"do", (PyCFunction) Py_source_do,
    METH_NOARGS, Py_source_do_doc},
  {"do_multi", (PyCFunction) Py_source_do_multi,
    METH_NOARGS, Py_source_do_multi_doc},
  {"close", (PyCFunction) Pyaubio_source_close,
    METH_NOARGS, Py_source_close_doc},
  {NULL} /* sentinel */
};

AUBIO_TYPEOBJECT(source, "aubio.source")
