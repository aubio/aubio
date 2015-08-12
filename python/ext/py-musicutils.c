#include "aubio-types.h"

PyObject *
Py_aubio_window(PyObject *self, PyObject *args)
{
  char_t *wintype = NULL;
  uint_t winlen = 0;
  fvec_t *window = NULL;

  if (!PyArg_ParseTuple (args, "|sI", &wintype, &winlen)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  window = new_aubio_window(wintype, winlen);
  if (window == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing window");
    return NULL;
  }

  return (PyObject *) PyAubio_CFvecToArray(window);
}

PyObject *
Py_aubio_level_lin(PyObject *self, PyObject *args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *level_lin;

  if (!PyArg_ParseTuple (args, "O:level_lin", &input)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);
  if (vec == NULL) {
    return NULL;
  }

  level_lin = Py_BuildValue("f", aubio_level_lin(vec));
  if (level_lin == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing level_lin");
    return NULL;
  }

  return level_lin;
}

PyObject *
Py_aubio_db_spl(PyObject *self, PyObject *args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *db_spl;

  if (!PyArg_ParseTuple (args, "O:db_spl", &input)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);
  if (vec == NULL) {
    return NULL;
  }

  db_spl = Py_BuildValue("f", aubio_db_spl(vec));
  if (db_spl == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing db_spl");
    return NULL;
  }

  return db_spl;
}

PyObject *
Py_aubio_silence_detection(PyObject *self, PyObject *args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *silence_detection;
  smpl_t threshold;

  if (!PyArg_ParseTuple (args, "Of:silence_detection", &input, &threshold)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);
  if (vec == NULL) {
    return NULL;
  }

  silence_detection = Py_BuildValue("I", aubio_silence_detection(vec, threshold));
  if (silence_detection == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing silence_detection");
    return NULL;
  }

  return silence_detection;
}

PyObject *
Py_aubio_level_detection(PyObject *self, PyObject *args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *level_detection;
  smpl_t threshold;

  if (!PyArg_ParseTuple (args, "Of:level_detection", &input, &threshold)) {
    PyErr_SetString (PyExc_ValueError, "failed parsing arguments");
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);
  if (vec == NULL) {
    return NULL;
  }

  level_detection = Py_BuildValue("f", aubio_level_detection(vec, threshold));
  if (level_detection == NULL) {
    PyErr_SetString (PyExc_ValueError, "failed computing level_detection");
    return NULL;
  }

  return level_detection;
}
