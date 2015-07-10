#ifndef _PY_AUBIO_MUSICUTILS_H_
#define _PY_AUBIO_MUSICUTILS_H_

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

#endif /* _PY_AUBIO_MUSICUTILS_H_ */
