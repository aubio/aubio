#include "aubio-types.h"

/* fvec type definition 

class fvec():
    def __init__(self, length = 1024, channels = 1):
        self.length = length 
        self.channels = channels 
        self.data = array(length, channels)

*/

static char Py_fvec_doc[] = "fvec object";

static PyObject *
Py_fvec_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int length= 0, channels = 0;
  Py_fvec *self;
  static char *kwlist[] = { "length", "channels", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &length, &channels)) {
    return NULL;
  }


  self = (Py_fvec *) type->tp_alloc (type, 0);

  self->length = Py_fvec_default_length;
  self->channels = Py_fvec_default_channels;

  if (self == NULL) {
    return NULL;
  }

  if (length > 0) {
    self->length = length;
  } else if (length < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of elements");
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
Py_fvec_init (Py_fvec * self, PyObject * args, PyObject * kwds)
{
  self->o = new_fvec (self->length, self->channels);
  if (self->o == NULL) {
    return -1;
  }

  return 0;
}

static void
Py_fvec_del (Py_fvec * self)
{
  del_fvec (self->o);
  self->ob_type->tp_free ((PyObject *) self);
}

static PyObject *
Py_fvec_repr (Py_fvec * self, PyObject * unused)
{
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyString_FromString ("aubio fvec of %d elements with %d channels");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("II", self->length, self->channels);
  if (args == NULL) {
    goto fail;
  }

  result = PyString_Format (format, args);

fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
}

static PyObject *
Py_fvec_print (Py_fvec * self, PyObject * unused)
{
  fvec_print (self->o);
  return Py_None;
}

Py_fvec *
PyAubio_ArrayToFvec (PyObject *input) {
  PyObject *array;
  Py_fvec *vec;
  uint_t i;
  // parsing input object into a Py_fvec
  if (PyObject_TypeCheck (input, &Py_fvecType)) {
    // input is an fvec, nothing else to do
    vec = (Py_fvec *) input;
  } else if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec 
    if (PyArray_NDIM (input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM (input) > 2) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than two dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT (input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
#if AUBIO_DO_CASTING
    } else if (PyArray_TYPE (input) != AUBIO_FLOAT) {
      // input data type is not float32, casting 
      array = PyArray_Cast ( (PyArrayObject*) input, AUBIO_FLOAT);
      if (array == NULL) {
        PyErr_SetString (PyExc_IndexError, "failed converting to NPY_FLOAT");
        goto fail;
      }
#else
    } else if (PyArray_TYPE (input) != AUBIO_FLOAT) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
#endif
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // create a new fvec object
    vec = (Py_fvec*) PyObject_New (Py_fvec, &Py_fvecType); 
    if (PyArray_NDIM (array) == 1) {
      vec->channels = 1;
      vec->length = PyArray_SIZE (array);
    } else {
      vec->channels = PyArray_DIM (array, 0);
      vec->length = PyArray_DIM (array, 1);
    }

    // no need to really allocate fvec, just its struct member 
    // vec->o = new_fvec (vec->length, vec->channels);
    vec->o = (fvec_t *)malloc(sizeof(fvec_t));
    vec->o->length = vec->length; vec->o->channels = vec->channels;
    vec->o->data = (smpl_t**)malloc(vec->o->channels * sizeof(smpl_t*));
    // hat data[i] point to array line
    for (i = 0; i < vec->channels; i++) {
      vec->o->data[i] = (smpl_t *) PyArray_GETPTR1 (array, i);
    }

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array or fvec as input");
    return NULL;
  }

  return vec;

fail:
  return NULL;
}

PyObject *
PyAubio_FvecToArray (Py_fvec * self)
{
  PyObject *array = NULL;
  if (self->channels == 1) {
    npy_intp dims[] = { self->length, 1 };
    array = PyArray_SimpleNewFromData (1, dims, NPY_FLOAT, self->o->data[0]);
  } else {
    uint_t i;
    npy_intp dims[] = { self->length, 1 };
    PyObject *concat = PyList_New (0), *tmp = NULL;
    for (i = 0; i < self->channels; i++) {
      tmp = PyArray_SimpleNewFromData (1, dims, NPY_FLOAT, self->o->data[i]);
      PyList_Append (concat, tmp);
      Py_DECREF (tmp);
    }
    array = PyArray_FromObject (concat, NPY_FLOAT, 2, 2);
    Py_DECREF (concat);
  }
  return array;
}

static Py_ssize_t
Py_fvec_getchannels (Py_fvec * self)
{
  return self->channels;
}

static PyObject *
Py_fvec_getitem (Py_fvec * self, Py_ssize_t index)
{
  PyObject *array;

  if (index < 0 || index >= self->channels) {
    PyErr_SetString (PyExc_IndexError, "no such channel");
    return NULL;
  }

  npy_intp dims[] = { self->length, 1 };
  array = PyArray_SimpleNewFromData (1, dims, NPY_FLOAT, self->o->data[index]);
  return array;
}

static int
Py_fvec_setitem (Py_fvec * self, Py_ssize_t index, PyObject * o)
{
  PyObject *array;

  if (index < 0 || index >= self->channels) {
    PyErr_SetString (PyExc_IndexError, "no such channel");
    return -1;
  }

  array = PyArray_FROM_OT (o, NPY_FLOAT);
  if (array == NULL) {
    PyErr_SetString (PyExc_ValueError, "should be an array of float");
    goto fail;
  }

  if (PyArray_NDIM (array) != 1) {
    PyErr_SetString (PyExc_ValueError, "should be a one-dimensional array");
    goto fail;
  }

  if (PyArray_SIZE (array) != self->length) {
    PyErr_SetString (PyExc_ValueError,
        "should be an array of same length as target fvec");
    goto fail;
  }

  self->o->data[index] = (smpl_t *) PyArray_GETPTR1 (array, 0);

  return 0;

fail:
  return -1;
}

static PyMemberDef Py_fvec_members[] = {
  // TODO remove READONLY flag and define getter/setter
  {"length", T_INT, offsetof (Py_fvec, length), READONLY,
      "length attribute"},
  {"channels", T_INT, offsetof (Py_fvec, channels), READONLY,
      "channels attribute"},
  {NULL}                        /* Sentinel */
};

static PyMethodDef Py_fvec_methods[] = {
  {"dump", (PyCFunction) Py_fvec_print, METH_NOARGS,
      "Dumps the contents of the vector to stdout."},
  {"__array__", (PyCFunction) PyAubio_FvecToArray, METH_NOARGS,
      "Returns the first channel as a numpy array."},
  {NULL}
};

static PySequenceMethods Py_fvec_tp_as_sequence = {
  (lenfunc) Py_fvec_getchannels,        /* sq_length         */
  0,                                    /* sq_concat         */
  0,                                    /* sq_repeat         */
  (ssizeargfunc) Py_fvec_getitem,       /* sq_item           */
  0,                                    /* sq_slice          */
  (ssizeobjargproc) Py_fvec_setitem,    /* sq_ass_item       */
  0,                                    /* sq_ass_slice      */
  0,                                    /* sq_contains       */
  0,                                    /* sq_inplace_concat */
  0,                                    /* sq_inplace_repeat */
};


PyTypeObject Py_fvecType = {
  PyObject_HEAD_INIT (NULL)
  0,                            /* ob_size           */
  "aubio.fvec",                 /* tp_name           */
  sizeof (Py_fvec),             /* tp_basicsize      */
  0,                            /* tp_itemsize       */
  (destructor) Py_fvec_del,     /* tp_dealloc        */
  0,                            /* tp_print          */
  0,                            /* tp_getattr        */
  0,                            /* tp_setattr        */
  0,                            /* tp_compare        */
  (reprfunc) Py_fvec_repr,      /* tp_repr           */
  0,                            /* tp_as_number      */
  &Py_fvec_tp_as_sequence,      /* tp_as_sequence    */
  0,                            /* tp_as_mapping     */
  0,                            /* tp_hash           */
  0,                            /* tp_call           */
  0,                            /* tp_str            */
  0,                            /* tp_getattro       */
  0,                            /* tp_setattro       */
  0,                            /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT,           /* tp_flags          */
  Py_fvec_doc,                  /* tp_doc            */
  0,                            /* tp_traverse       */
  0,                            /* tp_clear          */
  0,                            /* tp_richcompare    */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter           */
  0,                            /* tp_iternext       */
  Py_fvec_methods,              /* tp_methods        */
  Py_fvec_members,              /* tp_members        */
  0,                            /* tp_getset         */
  0,                            /* tp_base           */
  0,                            /* tp_dict           */
  0,                            /* tp_descr_get      */
  0,                            /* tp_descr_set      */
  0,                            /* tp_dictoffset     */
  (initproc) Py_fvec_init,      /* tp_init           */
  0,                            /* tp_alloc          */
  Py_fvec_new,                  /* tp_new            */
};
