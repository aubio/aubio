#ifndef PY_AUBIO_MUSICUTILS_H
#define PY_AUBIO_MUSICUTILS_H

static char Py_aubio_window_doc[] = ""
"window(string, integer) -> fvec\n"
"\n"
"Create a window\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> window('hanningz', 1024)\n"
"array([  0.00000000e+00,   9.41753387e-06,   3.76403332e-05, ...,\n"
"         8.46982002e-05,   3.76403332e-05,   9.41753387e-06], dtype=float32)";

PyObject * Py_aubio_window(PyObject *self, PyObject *args);

static char Py_aubio_level_lin_doc[] = ""
"level_lin(fvec) -> fvec\n"
"\n"
"Compute sound level on a linear scale.\n"
"\n"
"This gives the average of the square amplitudes.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> level_Lin(numpy.ones(1024))\n"
"1.0";

PyObject * Py_aubio_level_lin(PyObject *self, PyObject *args);

static char Py_aubio_db_spl_doc[] = ""
"Compute sound pressure level (SPL) in dB\n"
"\n"
"This quantity is often wrongly called 'loudness'.\n"
"\n"
"This gives ten times the log10 of the average of the square amplitudes.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> db_spl(numpy.ones(1024))\n"
"1.0";

PyObject * Py_aubio_db_spl(PyObject *self, PyObject *args);

static char Py_aubio_silence_detection_doc[] = ""
"Check if buffer level in dB SPL is under a given threshold\n"
"\n"
"Return 0 if level is under the given threshold, 1 otherwise.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> import numpy\n"""
">>> silence_detection(numpy.ones(1024, dtype=\"float32\"), -80)\n"
"0";

PyObject * Py_aubio_silence_detection(PyObject *self, PyObject *args);

static char Py_aubio_level_detection_doc[] = ""
"Get buffer level in dB SPL if over a given threshold, 1. otherwise.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> import numpy\n"""
">>> level_detection(0.7*numpy.ones(1024, dtype=\"float32\"), -80)\n"
"0";

PyObject * Py_aubio_level_detection(PyObject *self, PyObject *args);

static char Py_aubio_shift_doc[] = ""
"Swap left and right partitions of a vector\n"
"\n"
"Returns the swapped vector. The input vector is also modified.\n"
"\n"
"For a vector of length N, the partition is split at index N - N//2.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> import numpy\n"
">>> shift(numpy.arange(3, dtype=aubio.float_type))\n"
"array([2., 0., 1.], dtype=" AUBIO_NPY_SMPL_STR ")";
PyObject * Py_aubio_shift(PyObject *self, PyObject *args);

static char Py_aubio_ishift_doc[] = ""
"Swap right and left partitions of a vector\n"
"\n"
"Returns the swapped vector. The input vector is also modified.\n"
"\n"
"Unlike with shift(), the partition is split at index N//2.\n"
"\n"
"Example\n"
"-------\n"
"\n"
">>> import numpy\n"
">>> ishift(numpy.arange(3, dtype=aubio.float_type))\n"
"array([1., 2., 0.], dtype=" AUBIO_NPY_SMPL_STR ")";
PyObject * Py_aubio_ishift(PyObject *self, PyObject *args);

#endif /* PY_AUBIO_MUSICUTILS_H */
