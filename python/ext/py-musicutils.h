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

#endif /* _PY_AUBIO_MUSICUTILS_H_ */
