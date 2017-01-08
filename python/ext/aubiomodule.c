#define PY_AUBIO_MODULE_MAIN
#include "aubio-types.h"
#include "py-musicutils.h"

// this dummy macro is used to convince windows that a string passed as -D flag
// is just that, a string, and not a double.
#define REDEFINESTRING(x) #x
#define DEFINEDSTRING(x) REDEFINESTRING(x)

static char aubio_module_doc[] = "Python module for the aubio library";

static char Py_alpha_norm_doc[] = ""
"alpha_norm(fvec, integer) -> float\n"
"\n"
"Compute alpha normalisation factor on vector, given alpha\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> b = alpha_norm(a, 9)";

static char Py_bintomidi_doc[] = ""
"bintomidi(float, samplerate = integer, fftsize = integer) -> float\n"
"\n"
"Convert bin (float) to midi (float), given the sampling rate and the FFT size\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> midi = bintomidi(float, samplerate = 44100, fftsize = 1024)";

static char Py_miditobin_doc[] = ""
"miditobin(float, samplerate = integer, fftsize = integer) -> float\n"
"\n"
"Convert midi (float) to bin (float), given the sampling rate and the FFT size\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> bin = miditobin(midi, samplerate = 44100, fftsize = 1024)";

static char Py_bintofreq_doc[] = ""
"bintofreq(float, samplerate = integer, fftsize = integer) -> float\n"
"\n"
"Convert bin number (float) in frequency (Hz), given the sampling rate and the FFT size\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> freq = bintofreq(bin, samplerate = 44100, fftsize = 1024)";

static char Py_freqtobin_doc[] = ""
"freqtobin(float, samplerate = integer, fftsize = integer) -> float\n"
"\n"
"Convert frequency (Hz) in bin number (float), given the sampling rate and the FFT size\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> bin = freqtobin(freq, samplerate = 44100, fftsize = 1024)";

static char Py_zero_crossing_rate_doc[] = ""
"zero_crossing_rate(fvec) -> float\n"
"\n"
"Compute Zero crossing rate of a vector\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> z = zero_crossing_rate(a)";

static char Py_min_removal_doc[] = ""
"min_removal(fvec) -> float\n"
"\n"
"Remove the minimum value of a vector, in-place modification\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> min_removal(a)";

extern void add_ufuncs ( PyObject *m );
extern int generated_types_ready(void);

static PyObject *
Py_alpha_norm (PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t vec;
  smpl_t alpha;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "O" AUBIO_NPY_SMPL_CHR ":alpha_norm", &input, &alpha)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec(input, &vec)) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue (AUBIO_NPY_SMPL_CHR, fvec_alpha_norm (&vec, alpha));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static PyObject *
Py_bintomidi (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|" AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR , &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_bintomidi (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static PyObject *
Py_miditobin (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|" AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR , &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_miditobin (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static PyObject *
Py_bintofreq (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|" AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR, &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_bintofreq (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static PyObject *
Py_freqtobin (PyObject * self, PyObject * args)
{
  smpl_t input, samplerate, fftsize;
  smpl_t output;

  if (!PyArg_ParseTuple (args, "|" AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR AUBIO_NPY_SMPL_CHR, &input, &samplerate, &fftsize)) {
    return NULL;
  }

  output = aubio_freqtobin (input, samplerate, fftsize);

  return (PyObject *)PyFloat_FromDouble (output);
}

static PyObject *
Py_zero_crossing_rate (PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t vec;
  PyObject *result;

  if (!PyArg_ParseTuple (args, "O:zero_crossing_rate", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec(input, &vec)) {
    return NULL;
  }

  // compute the function
  result = Py_BuildValue (AUBIO_NPY_SMPL_CHR, aubio_zero_crossing_rate (&vec));
  if (result == NULL) {
    return NULL;
  }

  return result;
}

static PyObject *
Py_min_removal(PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t vec;

  if (!PyArg_ParseTuple (args, "O:min_removal", &input)) {
    return NULL;
  }

  if (input == NULL) {
    return NULL;
  }

  if (!PyAubio_ArrayToCFvec(input, &vec)) {
    return NULL;
  }

  // compute the function
  fvec_min_removal (&vec);

  // since this function does not return, we could return None
  //Py_RETURN_NONE;
  // however it is convenient to return the modified vector
  return (PyObject *) PyAubio_CFvecToArray(&vec);
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
  {"level_lin", Py_aubio_level_lin, METH_VARARGS, Py_aubio_level_lin_doc},
  {"db_spl", Py_aubio_db_spl, METH_VARARGS, Py_aubio_db_spl_doc},
  {"silence_detection", Py_aubio_silence_detection, METH_VARARGS, Py_aubio_silence_detection_doc},
  {"level_detection", Py_aubio_level_detection, METH_VARARGS, Py_aubio_level_detection_doc},
  {"window", Py_aubio_window, METH_VARARGS, Py_aubio_window_doc},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
// Python3 module definition
static struct PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
   "_aubio",          /* m_name */
   aubio_module_doc,  /* m_doc */
   -1,                /* m_size */
   aubio_methods,     /* m_methods */
   NULL,              /* m_reload */
   NULL,              /* m_traverse */
   NULL,              /* m_clear */
   NULL,              /* m_free */
};
#endif

void
aubio_log_function(int level, const char *message, void *data)
{
  // remove trailing \n
  char *pos;
  if ((pos=strchr(message, '\n')) != NULL) {
        *pos = '\0';
  }
  // warning or error
  if (level == AUBIO_LOG_ERR) {
    PyErr_Format(PyExc_RuntimeError, "%s", message);
  } else {
    PyErr_WarnEx(PyExc_UserWarning, message, 1);
  }
}

static PyObject *
initaubio (void)
{
  PyObject *m = NULL;
  int err;

  // fvec is defined in __init__.py
  if (   (PyType_Ready (&Py_cvecType) < 0)
      || (PyType_Ready (&Py_filterType) < 0)
      || (PyType_Ready (&Py_filterbankType) < 0)
      || (PyType_Ready (&Py_fftType) < 0)
      || (PyType_Ready (&Py_pvocType) < 0)
      || (PyType_Ready (&Py_sourceType) < 0)
      || (PyType_Ready (&Py_sinkType) < 0)
      // generated objects
      || (generated_types_ready() < 0 )
  ) {
    return m;
  }

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule3 ("_aubio", aubio_methods, aubio_module_doc);
#endif

  if (m == NULL) {
    return m;
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
  Py_INCREF (&Py_sinkType);
  PyModule_AddObject (m, "sink", (PyObject *) & Py_sinkType);

  PyModule_AddStringConstant(m, "float_type", AUBIO_NPY_SMPL_STR);
  PyModule_AddStringConstant(m, "__version__", DEFINEDSTRING(AUBIO_VERSION));

  // add generated objects
  add_generated_objects(m);

  // add ufunc
  add_ufuncs(m);

  aubio_log_set_level_function(AUBIO_LOG_ERR, aubio_log_function, NULL);
  aubio_log_set_level_function(AUBIO_LOG_WRN, aubio_log_function, NULL);
  return m;
}

#if PY_MAJOR_VERSION >= 3
    // Python3 init
    PyMODINIT_FUNC PyInit__aubio(void)
    {
        return initaubio();
    }
#else
    // Python 2 init
    PyMODINIT_FUNC init_aubio(void)
    {
        initaubio();
    }
#endif
