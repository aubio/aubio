# Examples build file

# Utility sources shared by all examples
utils_sources = files(
  'utils.c',
  'jackio.c',
)

# Declare jack dependency
jack = dependency('jack', required: false)
if jack.found()
  jack_dep = declare_dependency(link_args: jack.get_variable('link_args', default_value: []))
else
  jack_dep = declare_dependency()
endif

# Build utility objects library
utils_inc = include_directories('.')
src_inc = include_directories('../src')

# On Windows, link against static library; on other platforms use aubio_dep
if host_system == 'windows'
  utilsio_link_with = [libaubio_static]
  utilsio_deps = [jack_dep] + dependencies
else
  utilsio_link_with = []
  utilsio_deps = [aubio_dep] + (jack.found() ? [jack] : [])
endif

utilsio = static_library('utilsio',
  utils_sources,
  include_directories: [utils_inc, src_inc, config_inc],
  dependencies: utilsio_deps,
  link_with: utilsio_link_with,
  install: false,
)

# Example programs
example_programs = [
  'aubiomfcc',
  'aubionotes',
  'aubioonset',
  'aubiopitch',
  'aubioquiet',
  'aubiotrack',
]

# On Windows, need to include aubio_inc for headers
if host_system == 'windows'
  example_inc = [utils_inc, src_inc, config_inc, aubio_inc]
else
  example_inc = [utils_inc, src_inc, config_inc]
endif

# Build each example program
foreach prog : example_programs
  executable(prog,
    prog + '.c',
    include_directories: example_inc,
    link_with: [utilsio],
    dependencies: [],
    install: true,
  )
endforeach
