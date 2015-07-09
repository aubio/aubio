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
">>> window('hanningz', 1024)";

PyObject * Py_aubio_window(PyObject *self, PyObject *args);

#endif /* _PY_AUBIO_MUSICUTILS_H_ */
