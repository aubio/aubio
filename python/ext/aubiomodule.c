#define PY_AUBIO_MODULE_MAIN
#include "aubio-types.h"
#include "aubio-generated.h"

extern void add_generated_objects ( PyObject *m );
extern void add_ufuncs ( PyObject *m );
extern int generated_types_ready(void);

static char Py_alpha_norm_doc[] = "compute alpha normalisation factor";

static PyObject *
Py_alpha_norm (PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;
  smpl_t alpha;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "Of:alpha_norm", &input, &alpha)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue ("f", fvec_alpha_norm (vec, alpha));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static char Py_bintomidi_doc[] = "convert bin to midi";

static PyObject *
Py_bintomidi (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|fff", &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_bintomidi (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static char Py_miditobin_doc[] = "convert midi to bin";

static PyObject *
Py_miditobin (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|fff", &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_miditobin (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static char Py_bintofreq_doc[] = "convert bin to freq";

static PyObject *
Py_bintofreq (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|fff", &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_bintofreq (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static char Py_freqtobin_doc[] = "convert freq to bin";

static PyObject *
Py_freqtobin (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|fff", &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_freqtobin (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static char Py_zero_crossing_rate_doc[] = "compute zero crossing rate";

static PyObject *
Py_zero_crossing_rate (PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "O:zero_crossing_rate", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue ("f", aubio_zero_crossing_rate (vec));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static char Py_min_removal_doc[] = "compute zero crossing rate";

static PyObject *
Py_min_removal(PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;

  if (!PyArg_ParseTuple (args, "O:min_removal", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  // compute the function
  fvec_min_removal (vec);

  // since this function does not return, we could return None
  //Py_RETURN_NONE;
  // however it is convenient to return the modified vector
  return (PyObject *) PyAubio_CFvecToArray(vec);
  // or even without converting it back to an array
  //Py_INCREF(vec);
  //return (PyObject *)vec;
}

static PyMethodDef aubio_methods[] = {
  {"bintomidi", Py_bintomidi, METH_VARARGS, Py_bintomidi_doc},
  {"miditobin", Py_miditobin, METH_VARARGS, Py_miditobin_doc},
  {"bintofreq", Py_bintofreq, METH_VARARGS, Py_bintofreq_doc},
  {"freqtobin", Py_freqtobin, METH_VARARGS, Py_freqtobin_doc},
  {"alpha_norm", Py_alpha_norm, METH_VARARGS, Py_alpha_norm_doc},
  {"zero_crossing_rate", Py_zero_crossing_rate, METH_VARARGS, Py_zero_crossing_rate_doc},
  {"min_removal", Py_min_removal, METH_VARARGS, Py_min_removal_doc},
  {NULL, NULL} /* Sentinel */
};

static char aubio_module_doc[] = "Python module for the aubio library";

PyMODINIT_FUNC
init_aubio (void)
{
  PyObject *m;
  int err;

  // fvec is defined in __init__.py
  if (   (PyType_Ready (&Py_cvecType) < 0)
      || (PyType_Ready (&Py_filterType) < 0)
      || (PyType_Ready (&Py_filterbankType) < 0)
      || (PyType_Ready (&Py_fftType) < 0)
      || (PyType_Ready (&Py_pvocType) < 0)
      || (PyType_Ready (&Py_sourceType) < 0)
      // generated objects
      || (generated_types_ready() < 0 )
  ) {
    return;
  }

  m = Py_InitModule3 ("_aubio", aubio_methods, aubio_module_doc);

  if (m == NULL) {
    return;
  }

  err = _import_array ();
  if (err != 0) {
    fprintf (stderr,
        "Unable to import Numpy array from aubio module (error %d)\n", err);
  }

  Py_INCREF (&Py_cvecType);
  PyModule_AddObject (m, "cvec", (PyObject *) & Py_cvecType);
  Py_INCREF (&Py_filterType);
  PyModule_AddObject (m, "digital_filter", (PyObject *) & Py_filterType);
  Py_INCREF (&Py_filterbankType);
  PyModule_AddObject (m, "filterbank", (PyObject *) & Py_filterbankType);
  Py_INCREF (&Py_fftType);
  PyModule_AddObject (m, "fft", (PyObject *) & Py_fftType);
  Py_INCREF (&Py_pvocType);
  PyModule_AddObject (m, "pvoc", (PyObject *) & Py_pvocType);
  Py_INCREF (&Py_sourceType);
  PyModule_AddObject (m, "source", (PyObject *) & Py_sourceType);

  // add generated objects
  add_generated_objects(m);

  // add ufunc
  add_ufuncs(m);
}
