from distutils.core import setup, Extension

from os import listdir
generated_files = listdir('generated')
generated_files = filter(lambda x: x.endswith('.c'), generated_files)
generated_files = ['generated/'+f for f in generated_files]

setup(name="_aubio", version="1.0",
      ext_modules = [ 
        Extension("_aubio",
            ["aubiomodule.c",
            "py-fvec.c",
            "py-fmat.c",
            "py-cvec.c",
            "py-filter.c",
            # macroised 
            "py-filterbank.c",
            "py-fft.c",
            "py-phasevoc.c",
            # generated files
            ] + generated_files,
            include_dirs=['../../build/default/src', '../../src', '.' ],
            library_dirs=['../../build/default/src', '../../src/.libs' ],
            libraries=['aubio'])])

