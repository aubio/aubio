# Python bindings for aubio

py = import('python').find_installation('python3', modules: ['numpy'], required: true)

# Get C compiler for preprocessing (needed by gen_external.py)
cc = meson.get_compiler('c')

# Get numpy include directory
numpy_inc_result = run_command(py, 
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: false
)

if numpy_inc_result.returncode() == 0
  numpy_inc = include_directories(numpy_inc_result.stdout().strip())
else
  error('NumPy is required for building Python bindings. Please install numpy: pip install numpy')
endif

# Generate Python extension wrapper sources
# This generates gen-*.c files for each aubio object (dct, mfcc, notes, etc.)
# and aubio-generated.c/.h that ties them all together
gen_external_script = files('lib/gen_external.py')

gen_wrapper_sources = custom_target('generate_python_wrappers',
  output: [
    'aubio-generated.c',
    'aubio-generated.h',
    'gen-dct.c',
    'gen-mfcc.c',
    'gen-notes.c',
    'gen-onset.c',
    'gen-pitch.c',
    'gen-pitchshift.c',
    'gen-sampler.c',
    'gen-specdesc.c',
    'gen-tempo.c',
    'gen-tss.c',
    'gen-wavetable.c',
  ],
  command: [
    py,
    gen_external_script,
    meson.project_source_root() / 'src' / 'aubio.h',
    '@OUTDIR@',
  ],
  depend_files: gen_external_script,
  # Set CC environment variable so gen_external.py can find the compiler
  env: {'CC': ' '.join(cc.cmd_array())},
)

# Python extension sources (only hand-written files)
pyaubio_sources = files(
  'ext/aubiomodule.c',
  'ext/aubioproxy.c',
  'ext/py-cvec.c',
  'ext/py-fft.c',
  'ext/py-filter.c',
  'ext/py-filterbank.c',
  'ext/py-musicutils.c',
  'ext/py-phasevoc.c',
  'ext/py-sink.c',
  'ext/py-source.c',
  'ext/ufuncs.c',
)

# Add all generated wrapper sources
pyaubio_sources += gen_wrapper_sources

# Include directories for Python extension
# The current build directory ('.') is included to find generated aubio-generated.h
pyaubio_inc = [include_directories('ext', '../src', '.')]

# Determine if we should build using installed libaubio or embed it
# For now, we'll link against the libaubio we just built
pyaubio_c_args = [
  '-DHAVE_CONFIG_H',
  '-DUSE_LOCAL_AUBIO',
  '-DAUBIO_VERSION=' + aubio_version,
]

# Platform-specific args and dependencies
if host_system == 'darwin'
  pyaubio_link_args = [
    '-framework', 'CoreFoundation',
    '-framework', 'AudioToolbox',
  ]
elif host_system == 'linux'
  # On Linux, when linking C++ static libraries (rubberband) and pthread-using
  # libraries (FFmpeg) into a shared Python extension, we must add these as
  # dependencies (not just link flags) to ensure proper link order
  thread_dep = dependency('threads', required: true)
  dependencies += [thread_dep]
  
  # Add C++ standard library as a dependency
  # Use the C compiler to find libstdc++ (no need for C++ compiler in project)
  c_compiler = meson.get_compiler('c')
  cpp_stdlib = c_compiler.find_library('stdc++', required: true)
  dependencies += [cpp_stdlib]
  
  pyaubio_link_args = []
else
  pyaubio_link_args = []
endif

# Build Python extension module
# For portable wheels, always link against the static library
# This bundles all aubio code into the Python extension
pyaubio_link_with = libaubio_static
pyaubio_deps = [py.dependency()]
# Add aubio's dependencies directly since we're linking statically
pyaubio_deps += dependencies

py.extension_module('_aubio',
  pyaubio_sources,
  include_directories: [pyaubio_inc, numpy_inc, config_inc],
  dependencies: pyaubio_deps,
  c_args: pyaubio_c_args,
  link_args: pyaubio_link_args,
  link_with: pyaubio_link_with,
  install: true,
  subdir: 'aubio',
)

# Install Python package files
py.install_sources(
  'lib/aubio/__init__.py',
  'lib/aubio/cmd.py',
  'lib/aubio/cut.py',
  'lib/aubio/midiconv.py',
  'lib/aubio/slicing.py',
  subdir: 'aubio',
  pure: false,
)
