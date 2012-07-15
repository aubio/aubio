#include "aubio-types.h"

#define AUBIO_DECLARE(NAME, PARAMS...) \
typedef struct { \
  PyObject_HEAD \
  aubio_ ## NAME ## _t * o; \
  PARAMS; \
} Py_## NAME;

#define AUBIO_INIT(NAME, PARAMS... ) \
static int \
Py_ ## NAME ## _init (Py_ ## NAME * self, PyObject * args, PyObject * kwds) \
{ \
  self->o = new_aubio_## NAME ( PARAMS ); \
  if (self->o == NULL) { \
    PyErr_SetString (PyExc_StandardError, "error creating object"); \
    return -1; \
  } \
\
  return 0; \
}

#define AUBIO_DEL(NAME) \
static void \
Py_ ## NAME ## _del ( Py_ ## NAME * self) \
{ \
  del_aubio_ ## NAME (self->o); \
  self->ob_type->tp_free ((PyObject *) self); \
}

#define AUBIO_MEMBERS_START(NAME) \
static PyMemberDef Py_ ## NAME ## _members[] = {

#define AUBIO_MEMBERS_STOP(NAME) \
  {NULL} \
};

#define AUBIO_METHODS(NAME) \
static PyMethodDef Py_ ## NAME ## _methods[] = { \
  {NULL} \
};


#define AUBIO_TYPEOBJECT(NAME, PYNAME) \
PyTypeObject Py_ ## NAME ## Type = { \
  PyObject_HEAD_INIT (NULL)    \
  0,                           \
  PYNAME,                      \
  sizeof (Py_ ## NAME),          \
  0,                           \
  (destructor) Py_ ## NAME ## _del,  \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  (ternaryfunc)Py_ ## NAME ## _do,   \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  Py_TPFLAGS_DEFAULT,          \
  Py_ ## NAME ## _doc,               \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  Py_ ## NAME ## _methods,           \
  Py_ ## NAME ## _members,           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  0,                           \
  (initproc) Py_ ## NAME ## _init,   \
  0,                           \
  Py_ ## NAME ## _new,               \
};

// some more helpers
#define AUBIO_NEW_VEC(name, type, lengthval) \
  name = (type *) PyObject_New (type, & type ## Type); \
  name->length = lengthval;
