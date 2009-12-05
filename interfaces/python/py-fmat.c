#include "aubio-types.h"

/* fmat type definition 

class fmat():
    def __init__(self, length = 1024, height = 1):
        self.length = length 
        self.height = height 
        self.data = array(length, height)

*/

static char Py_fmat_doc[] = "fmat object";

static PyObject *
Py_fmat_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int length= 0, height = 0;
  Py_fmat *self;
  static char *kwlist[] = { "length", "height", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|II", kwlist,
          &length, &height)) {
    return NULL;
  }


  self = (Py_fmat *) type->tp_alloc (type, 0);

  self->length = Py_default_vector_length;
  self->height = Py_default_vector_height;

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

  if (height > 0) {
    self->height = height;
  } else if (height < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of height");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_fmat_init (Py_fmat * self, PyObject * args, PyObject * kwds)
{
  self->o = new_fmat (self->length, self->height);
  if (self->o == NULL) {
    return -1;
  }

  return 0;
}

static void
Py_fmat_del (Py_fmat * self)
{
  del_fmat (self->o);
  self->ob_type->tp_free ((PyObject *) self);
}

static PyObject *
Py_fmat_repr (Py_fmat * self, PyObject * unused)
{
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyString_FromString ("aubio fmat of %d elements with %d height");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("II", self->length, self->height);
  if (args == NULL) {
    goto fail;
  }
  fmat_print ( self->o );

  result = PyString_Format (format, args);

fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
}

Py_fmat *
PyAubio_ArrayTofmat (PyObject *input) {
  PyObject *array;
  Py_fmat *vec;
  uint_t i;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  // parsing input object into a Py_fmat
  if (PyObject_TypeCheck (input, &Py_fmatType)) {
    // input is an fmat, nothing else to do
    vec = (Py_fmat *) input;
  } else if (PyArray_Check(input)) {

    // we got an array, convert it to an fmat 
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
    } else if (PyArray_TYPE (input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // create a new fmat object
    vec = (Py_fmat*) PyObject_New (Py_fmat, &Py_fmatType); 
    if (PyArray_NDIM (array) == 1) {
      vec->height = 1;
      vec->length = PyArray_SIZE (array);
    } else {
      vec->height = PyArray_DIM (array, 0);
      vec->length = PyArray_DIM (array, 1);
    }

    // no need to really allocate fmat, just its struct member 
    // vec->o = new_fmat (vec->length, vec->height);
    vec->o = (fmat_t *)malloc(sizeof(fmat_t));
    vec->o->length = vec->length; vec->o->height = vec->height;
    vec->o->data = (smpl_t**)malloc(vec->o->height * sizeof(smpl_t*));
    // hat data[i] point to array line
    for (i = 0; i < vec->height; i++) {
      vec->o->data[i] = (smpl_t *) PyArray_GETPTR1 (array, i);
    }

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array or fmat as input");
    return NULL;
  }

  return vec;

fail:
  return NULL;
}

PyObject *
PyAubio_CfmatToArray (fmat_t * self)
{
  PyObject *array = NULL;
  uint_t i;
  npy_intp dims[] = { self->length, 1 };
  PyObject *concat = PyList_New (0), *tmp = NULL;
  for (i = 0; i < self->height; i++) {
    tmp = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->data[i]);
    PyList_Append (concat, tmp);
    Py_DECREF (tmp);
  }
  array = PyArray_FromObject (concat, AUBIO_NPY_SMPL, 2, 2);
  Py_DECREF (concat);
  return array;
}

PyObject *
PyAubio_FmatToArray (Py_fmat * self)
{
  PyObject *array = NULL;
  if (self->height == 1) {
    npy_intp dims[] = { self->length, 1 };
    array = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->o->data[0]);
  } else {
    uint_t i;
    npy_intp dims[] = { self->length, 1 };
    PyObject *concat = PyList_New (0), *tmp = NULL;
    for (i = 0; i < self->height; i++) {
      tmp = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->o->data[i]);
      PyList_Append (concat, tmp);
      Py_DECREF (tmp);
    }
    array = PyArray_FromObject (concat, AUBIO_NPY_SMPL, 2, 2);
    Py_DECREF (concat);
  }
  return array;
}

static Py_ssize_t
Py_fmat_get_height (Py_fmat * self)
{
  return self->height;
}

static PyObject *
Py_fmat_getitem (Py_fmat * self, Py_ssize_t index)
{
  PyObject *array;

  if (index < 0 || index >= self->height) {
    PyErr_SetString (PyExc_IndexError, "no such channel");
    return NULL;
  }

  npy_intp dims[] = { self->length, 1 };
  array = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->o->data[index]);
  return array;
}

static int
Py_fmat_setitem (Py_fmat * self, Py_ssize_t index, PyObject * o)
{
  PyObject *array;

  if (index < 0 || index >= self->height) {
    PyErr_SetString (PyExc_IndexError, "no such channel");
    return -1;
  }

  array = PyArray_FROM_OT (o, AUBIO_NPY_SMPL);
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
        "should be an array of same length as target fmat");
    goto fail;
  }

  self->o->data[index] = (smpl_t *) PyArray_GETPTR1 (array, 0);

  return 0;

fail:
  return -1;
}

static PyMemberDef Py_fmat_members[] = {
  // TODO remove READONLY flag and define getter/setter
  {"length", T_INT, offsetof (Py_fmat, length), READONLY,
      "length attribute"},
  {"height", T_INT, offsetof (Py_fmat, height), READONLY,
      "height attribute"},
  {NULL}                        /* Sentinel */
};

static PyMethodDef Py_fmat_methods[] = {
  {"__array__", (PyCFunction) PyAubio_FmatToArray, METH_NOARGS,
      "Returns the vector as a numpy array."},
  {NULL}
};

static PySequenceMethods Py_fmat_tp_as_sequence = {
  (lenfunc) Py_fmat_get_height,        /* sq_length         */
  0,                                    /* sq_concat         */
  0,                                    /* sq_repeat         */
  (ssizeargfunc) Py_fmat_getitem,       /* sq_item           */
  0,                                    /* sq_slice          */
  (ssizeobjargproc) Py_fmat_setitem,    /* sq_ass_item       */
  0,                                    /* sq_ass_slice      */
  0,                                    /* sq_contains       */
  0,                                    /* sq_inplace_concat */
  0,                                    /* sq_inplace_repeat */
};


PyTypeObject Py_fmatType = {
  PyObject_HEAD_INIT (NULL)
  0,                            /* ob_size           */
  "aubio.fmat",                 /* tp_name           */
  sizeof (Py_fmat),             /* tp_basicsize      */
  0,                            /* tp_itemsize       */
  (destructor) Py_fmat_del,     /* tp_dealloc        */
  0,                            /* tp_print          */
  0,                            /* tp_getattr        */
  0,                            /* tp_setattr        */
  0,                            /* tp_compare        */
  (reprfunc) Py_fmat_repr,      /* tp_repr           */
  0,                            /* tp_as_number      */
  &Py_fmat_tp_as_sequence,      /* tp_as_sequence    */
  0,                            /* tp_as_mapping     */
  0,                            /* tp_hash           */
  0,                            /* tp_call           */
  0,                            /* tp_str            */
  0,                            /* tp_getattro       */
  0,                            /* tp_setattro       */
  0,                            /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT,           /* tp_flags          */
  Py_fmat_doc,                  /* tp_doc            */
  0,                            /* tp_traverse       */
  0,                            /* tp_clear          */
  0,                            /* tp_richcompare    */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter           */
  0,                            /* tp_iternext       */
  Py_fmat_methods,              /* tp_methods        */
  Py_fmat_members,              /* tp_members        */
  0,                            /* tp_getset         */
  0,                            /* tp_base           */
  0,                            /* tp_dict           */
  0,                            /* tp_descr_get      */
  0,                            /* tp_descr_set      */
  0,                            /* tp_dictoffset     */
  (initproc) Py_fmat_init,      /* tp_init           */
  0,                            /* tp_alloc          */
  Py_fmat_new,                  /* tp_new            */
};
