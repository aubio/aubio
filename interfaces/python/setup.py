from distutils.core import setup, Extension

setup(name="_aubio", version="1.0",
      ext_modules = [ 
        Extension("_aubio",
            ["aubiomodule.c", "py-fvec.c"],
            include_dirs=['../../build/default/src', '../../src' ],
            library_dirs=['../../build/default/src', '../../src/.libs' ],
            libraries=['aubio'])])

