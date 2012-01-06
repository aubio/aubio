from distutils.core import setup, Extension

from os import listdir
generated_files = listdir('generated')
generated_files = filter(lambda x: x.endswith('.c'), generated_files)
generated_files = ['generated/'+f for f in generated_files]

setup(name="_aubio", version="1.0",
      ext_modules = [ 
        Extension("_aubio",
            ["aubiomodule.c",
            "aubioproxy.c",
            "py-cvec.c",
            # example without macro
            "py-filter.c",
            # macroised
            "py-filterbank.c",
            "py-fft.c",
            "py-phasevoc.c",
            # generated files
            ] + generated_files,
            include_dirs=['../../build/src', '../../src', '.' ],
            library_dirs=['../../build/src', '../../src/.libs' ],
            libraries=['aubio'])])

