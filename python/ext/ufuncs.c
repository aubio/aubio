#define PY_AUBIO_MODULE_UFUNC
#include "aubio-types.h"

static void unwrap2pi(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        *((double *)out) = aubio_unwrap2pi(*(double *)in);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void unwrap2pif(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        *((float *)out) = aubio_unwrap2pi(*(float *)in);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static char Py_unwrap2pi_doc[] = "map angle to unit circle [-pi, pi[";

PyUFuncGenericFunction unwrap2pi_functions[] = {
  &unwrap2pif, &unwrap2pi,
  //PyUFunc_f_f_As_d_d, PyUFunc_d_d,
  //PyUFunc_g_g, PyUFunc_OO_O_method,
};

static void* unwrap2pi_data[] = {
  (void *)aubio_unwrap2pi,
  (void *)aubio_unwrap2pi,
  //(void *)unwrap2pil,
  //(void *)unwrap2pio,
};

static char unwrap2pi_types[] = {
  NPY_FLOAT, NPY_FLOAT,
  NPY_DOUBLE, NPY_DOUBLE,
  //NPY_LONGDOUBLE, NPY_LONGDOUBLE,
  //NPY_OBJECT, NPY_OBJECT,
};

void add_ufuncs ( PyObject *m )
{
  int err = 0;

  err = _import_umath ();
  if (err != 0) {
    fprintf (stderr,
        "Unable to import Numpy umath from aubio module (error %d)\n", err);
  }

  PyObject *f, *dict;
  dict = PyModule_GetDict(m);
  f = PyUFunc_FromFuncAndData(unwrap2pi_functions,
          unwrap2pi_data, unwrap2pi_types, 2, 1, 1,
          PyUFunc_None, "unwrap2pi", Py_unwrap2pi_doc, 0);
  PyDict_SetItemString(dict, "unwrap2pi", f);
  Py_DECREF(f);

  return;
}
