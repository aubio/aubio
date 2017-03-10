#include "aubio-types.h"

typedef struct
{
  PyObject_HEAD
  aubio_source_t * o;
  char_t* uri;
  uint_t samplerate;
  uint_t channels;
  uint_t hop_size;
  uint_t duration;
  PyObject *read_to;
  fvec_t c_read_to;
  PyObject *mread_to;
  fmat_t c_mread_to;
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

static char Py_source_seek_doc[] = ""
"x.seek(position)\n"
"\n"
"Seek to resampled frame position.";

static PyObject *
Py_source_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
  Py_source *self;
  char_t* uri = NULL;
  uint_t samplerate = 0;
  uint_t hop_size = 0;
  uint_t channels = 0;
  static char *kwlist[] = { "uri", "samplerate", "hop_size", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|sIII", kwlist,
          &uri, &samplerate, &hop_size, &channels)) {
    return NULL;
  }

  self = (Py_source *) pytype->tp_alloc (pytype, 0);

  if (self == NULL) {
    return NULL;
  }

  self->uri = NULL;
  if (uri != NULL) {
    self->uri = (char_t *)malloc(sizeof(char_t) * (strnlen(uri, PATH_MAX) + 1));
    strncpy(self->uri, uri, strnlen(uri, PATH_MAX) + 1);
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

  self->channels = 1;
  if ((sint_t)channels >= 0) {
    self->channels = channels;
  } else if ((sint_t)channels < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative value for channels");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_source_init (Py_source * self, PyObject * args, PyObject * kwds)
{
  self->o = new_aubio_source ( self->uri, self->samplerate, self->hop_size );
  if (self->o == NULL) {
    // PyErr_Format(PyExc_RuntimeError, ...) was set above by new_ which called
    // AUBIO_ERR when failing
    return -1;
  }
  self->samplerate = aubio_source_get_samplerate ( self->o );
  if (self->channels == 0) {
    self->channels = aubio_source_get_channels ( self->o );
  }
  self->duration = aubio_source_get_duration ( self->o );

  self->read_to = new_py_fvec(self->hop_size);
  self->mread_to = new_py_fmat(self->channels, self->hop_size);

  return 0;
}

static void
Py_source_del (Py_source *self, PyObject *unused)
{
  if (self->o) {
    del_aubio_source(self->o);
    free(self->c_mread_to.data);
  }
  if (self->uri) {
    free(self->uri);
  }
  Py_XDECREF(self->read_to);
  Py_XDECREF(self->mread_to);
  Py_TYPE(self)->tp_free((PyObject *) self);
}


/* function Py_source_do */
static PyObject *
Py_source_do(Py_source * self, PyObject * args)
{
  PyObject *outputs;
  uint_t read;
  read = 0;

  Py_INCREF(self->read_to);
  if (!PyAubio_ArrayToCFvec(self->read_to, &(self->c_read_to))) {
    return NULL;
  }
  /* compute _do function */
  aubio_source_do (self->o, &(self->c_read_to), &read);

  outputs = PyTuple_New(2);
  PyTuple_SetItem( outputs, 0, self->read_to );
  PyTuple_SetItem( outputs, 1, (PyObject *)PyLong_FromLong(read));
  return outputs;
}

/* function Py_source_do_multi */
static PyObject *
Py_source_do_multi(Py_source * self, PyObject * args)
{
  PyObject *outputs;
  uint_t read;
  read = 0;

  Py_INCREF(self->mread_to);
  if (!PyAubio_ArrayToCFmat(self->mread_to,  &(self->c_mread_to))) {
    return NULL;
  }
  /* compute _do function */
  aubio_source_do_multi (self->o, &(self->c_mread_to), &read);

  outputs = PyTuple_New(2);
  PyTuple_SetItem( outputs, 0, self->mread_to);
  PyTuple_SetItem( outputs, 1, (PyObject *)PyLong_FromLong(read));
  return outputs;
}

static PyMemberDef Py_source_members[] = {
  {"uri", T_STRING, offsetof (Py_source, uri), READONLY,
    "path at which the source was created"},
  {"samplerate", T_INT, offsetof (Py_source, samplerate), READONLY,
    "samplerate at which the source is viewed"},
  {"channels", T_INT, offsetof (Py_source, channels), READONLY,
    "number of channels found in the source"},
  {"hop_size", T_INT, offsetof (Py_source, hop_size), READONLY,
    "number of consecutive frames that will be read at each do or do_multi call"},
  {"duration", T_INT, offsetof (Py_source, duration), READONLY,
    "total number of frames in the source (estimated)"},
  { NULL } // sentinel
};

static PyObject *
Pyaubio_source_get_samplerate (Py_source *self, PyObject *unused)
{
  uint_t tmp = aubio_source_get_samplerate (self->o);
  return (PyObject *)PyLong_FromLong (tmp);
}

static PyObject *
Pyaubio_source_get_channels (Py_source *self, PyObject *unused)
{
  uint_t tmp = aubio_source_get_channels (self->o);
  return (PyObject *)PyLong_FromLong (tmp);
}

static PyObject *
Pyaubio_source_close (Py_source *self, PyObject *unused)
{
  if (aubio_source_close(self->o) != 0) return NULL;
  Py_RETURN_NONE;
}

static PyObject *
Pyaubio_source_seek (Py_source *self, PyObject *args)
{
  uint_t err = 0;

  int position;
  if (!PyArg_ParseTuple (args, "I", &position)) {
    return NULL;
  }

  if (position < 0) {
    PyErr_Format(PyExc_ValueError,
        "error when seeking in source: can not seek to negative value %d",
        position);
    return NULL;
  }

  err = aubio_source_seek(self->o, position);
  if (err != 0) {
    PyErr_SetString (PyExc_ValueError,
        "error when seeking in source");
    return NULL;
  }
  Py_RETURN_NONE;
}

static char Pyaubio_source_enter_doc[] = "";
static PyObject* Pyaubio_source_enter(Py_source *self, PyObject *unused) {
  Py_INCREF(self);
  return (PyObject*)self;
}

static char Pyaubio_source_exit_doc[] = "";
static PyObject* Pyaubio_source_exit(Py_source *self, PyObject *unused) {
  return Pyaubio_source_close(self, unused);
}

static PyObject* Pyaubio_source_iter(PyObject *self) {
  Py_INCREF(self);
  return (PyObject*)self;
}

static PyObject* Pyaubio_source_iter_next(Py_source *self) {
  PyObject *done, *size;
  if (self->channels == 1) {
    done = Py_source_do(self, NULL);
  } else {
    done = Py_source_do_multi(self, NULL);
  }
  if (!PyTuple_Check(done)) {
    PyErr_Format(PyExc_ValueError,
        "error when reading source: not opened?");
    return NULL;
  }
  size = PyTuple_GetItem(done, 1);
  if (size != NULL && PyLong_Check(size)) {
    if (PyLong_AsLong(size) == (long)self->hop_size) {
      PyObject *vec = PyTuple_GetItem(done, 0);
      return vec;
    } else if (PyLong_AsLong(size) > 0) {
      // short read, return a shorter array
      PyArrayObject *shortread = (PyArrayObject*)PyTuple_GetItem(done, 0);
      PyArray_Dims newdims;
      PyObject *reshaped;
      newdims.len = PyArray_NDIM(shortread);
      newdims.ptr = PyArray_DIMS(shortread);
      // mono or multiple channels?
      if (newdims.len == 1) {
        newdims.ptr[0] = PyLong_AsLong(size);
      } else {
        newdims.ptr[1] = PyLong_AsLong(size);
      }
      reshaped = PyArray_Newshape(shortread, &newdims, NPY_CORDER);
      Py_DECREF(shortread);
      return reshaped;
    } else {
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
  } else {
    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
  }
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
  {"seek", (PyCFunction) Pyaubio_source_seek,
    METH_VARARGS, Py_source_seek_doc},
  {"__enter__", (PyCFunction)Pyaubio_source_enter, METH_NOARGS,
    Pyaubio_source_enter_doc},
  {"__exit__",  (PyCFunction)Pyaubio_source_exit, METH_VARARGS,
    Pyaubio_source_exit_doc},
  {NULL} /* sentinel */
};

PyTypeObject Py_sourceType = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "aubio.source",
  sizeof (Py_source),
  0,
  (destructor) Py_source_del,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  (ternaryfunc)Py_source_do,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  Py_source_doc,
  0,
  0,
  0,
  0,
  Pyaubio_source_iter,
  (unaryfunc)Pyaubio_source_iter_next,
  Py_source_methods,
  Py_source_members,
  0,
  0,
  0,
  0,
  0,
  0,
  (initproc) Py_source_init,
  0,
  Py_source_new,
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
