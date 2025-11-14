project('aubio', 'c',
  version: '0.5.0-alpha',
  license: 'GPL-3.0-or-later',
  default_options: [
    'warning_level=2',
    'c_std=c99',
  ],
  meson_version: '>=1.9.0',
)

# Parse version from VERSION file
fs = import('fs')
version_file_path = meson.project_source_root() / 'VERSION'
version_contents = run_command(
  find_program('python3', 'python'),
  '-c',
  'import sys; print(open(sys.argv[1]).read())',
  version_file_path,
  check: true
).stdout().strip()

version_data = {}
foreach line : version_contents.split('\n')
  if line.contains('=')
    parts = line.split('=')
    key = parts[0]
    value = parts[1]
    if key == 'AUBIO_MAJOR_VERSION'
      version_data += {'major': value.to_int()}
    elif key == 'AUBIO_MINOR_VERSION'
      version_data += {'minor': value.to_int()}
    elif key == 'AUBIO_PATCH_VERSION'
      version_data += {'patch': value.to_int()}
    elif key == 'AUBIO_VERSION_STATUS'
      version_data += {'status': value.strip().strip('\'"')}
    elif key == 'LIBAUBIO_LT_CUR'
      version_data += {'lt_cur': value.to_int()}
    elif key == 'LIBAUBIO_LT_REV'
      version_data += {'lt_rev': value.to_int()}
    elif key == 'LIBAUBIO_LT_AGE'
      version_data += {'lt_age': value.to_int()}
    endif
  endif
endforeach

aubio_version = '@0@.@1@.@2@@3@'.format(
  version_data.get('major'),
  version_data.get('minor'),
  version_data.get('patch'),
  version_data.get('status', '')
)

# Library version (soversion)
lib_version = '@0@.@1@.@2@'.format(
  version_data.get('lt_cur'),
  version_data.get('lt_rev'),
  version_data.get('lt_age')
)

message('Building aubio version: ' + aubio_version)
message('Library version: ' + lib_version)

# Compiler setup
cc = meson.get_compiler('c')
host_system = host_machine.system()

# Ensure POSIX feature macros expose prototypes like strerror_r on glibc
if host_system != 'windows'
  add_project_arguments('-D_POSIX_C_SOURCE=200112L', language: 'c')
endif

# Security hardening compiler flags
# These flags add runtime protections and compile-time checks
# Can be disabled with -Dsecurity_hardening=false if needed for compatibility
if get_option('security_hardening')
  security_flags = []

  # Stack protection - prevents stack buffer overflows
  if cc.has_argument('-fstack-protector-strong')
    security_flags += ['-fstack-protector-strong']
  elif cc.has_argument('-fstack-protector')
    security_flags += ['-fstack-protector']
  endif

  # Fortify source - adds buffer overflow checks for standard library functions
  # Only works in optimized builds (-O1 or higher)
  if get_option('buildtype') != 'plain' and get_option('buildtype') != 'debug'
    if cc.has_argument('-D_FORTIFY_SOURCE=2')
      security_flags += ['-D_FORTIFY_SOURCE=2']
    endif
  endif

  # Format string security - warns about potentially insecure format strings
  if cc.has_argument('-Wformat')
    security_flags += ['-Wformat']
  endif
  if cc.has_argument('-Wformat-security')
    security_flags += ['-Wformat-security']
  endif

  # Treat implicit function declarations as errors (security issue)
  if cc.has_argument('-Werror=implicit-function-declaration')
    security_flags += ['-Werror=implicit-function-declaration']
  endif

  # Additional hardening warnings
  if cc.has_argument('-Warray-bounds')
    security_flags += ['-Warray-bounds']
  endif

  if security_flags.length() > 0
    add_project_arguments(security_flags, language: 'c')
    message('Security hardening flags enabled: ' + ' '.join(security_flags))
  endif
else
  message('Security hardening disabled')
endif

# Position-independent code for shared libraries (security best practice)
# PIE (Position Independent Executable) is handled separately for executables via b_pie option
# Note: -fPIC is automatically handled by meson for shared libraries
# We don't add -pie here as it will be applied at link time for executables only

# vcpkg integration for cross-platform dependency management
# Note: In CI/CD, vcpkg dependencies are pre-installed by cibuildwheel before-all step
# Check if dependencies are already installed before running vcpkg
vcpkg_install_root = meson.project_source_root() / 'vcpkg_installed'

# Determine vcpkg triplet based on host system and architecture
# Must match the triplet used in CI/CD (see pyproject.toml [tool.cibuildwheel] sections)
vcpkg_triplet = 'unknown'
if host_system == 'windows'
  # Windows: Use x64-windows-release triplet (release-only, dynamic linking)
  # This matches pyproject.toml [tool.cibuildwheel.windows] before-all step
  if host_machine.cpu_family() == 'x86_64'
    vcpkg_triplet = 'x64-windows-release'
  elif host_machine.cpu_family() == 'x86'
    vcpkg_triplet = 'x86-windows-release'
  elif host_machine.cpu_family() == 'aarch64'
    vcpkg_triplet = 'arm64-windows-release'
  else
    vcpkg_triplet = 'x64-windows-release'
  endif
elif host_system == 'darwin'
  if host_machine.cpu_family() == 'aarch64'
    vcpkg_triplet = 'arm64-osx'
  else
    vcpkg_triplet = 'x64-osx'
  endif
elif host_system == 'linux'
  # Linux: Use custom triplets with -fPIC for all architectures
  # This matches pyproject.toml [tool.cibuildwheel.linux] before-all step
  if host_machine.cpu_family() == 'aarch64'
    vcpkg_triplet = 'arm64-linux-pic'
  else
    vcpkg_triplet = 'x64-linux-pic'
  endif
endif

message('Using vcpkg triplet: ' + vcpkg_triplet)

# Check if vcpkg dependencies are already installed (e.g., by CI/CD before-all step)
vcpkg_installed_dir = vcpkg_install_root / vcpkg_triplet
vcpkg_already_installed = fs.is_dir(vcpkg_installed_dir)

if vcpkg_already_installed
  message('vcpkg dependencies already installed at: ' + vcpkg_installed_dir)
  vcpkg_found = true
else
  # Dependencies not pre-installed, try to install them now
  vcpkg_exe = find_program('vcpkg', required: false)
  if vcpkg_exe.found()
    message('vcpkg found, installing dependencies from vcpkg.json manifest...')
    
    # Install vcpkg dependencies with custom triplets for PIC support
    vcpkg_result = run_command(
      vcpkg_exe,
      'install',
      '--triplet=' + vcpkg_triplet,
      '--overlay-triplets=' + meson.project_source_root() / 'vcpkg-triplets',
      '--x-wait-for-lock',
      '--x-manifest-root=' + meson.project_source_root(),
      '--x-install-root=' + vcpkg_install_root,
      check: false
    )
    
    if vcpkg_result.returncode() == 0
      message('vcpkg dependencies installed successfully')
      vcpkg_found = true
    else
      warning('vcpkg install failed, falling back to system dependencies')
      message(vcpkg_result.stderr())
      vcpkg_found = false
    endif
  else
    message('vcpkg not found, using system dependencies')
    vcpkg_found = false
  endif
endif

# If vcpkg dependencies are available (either pre-installed or just installed), set up paths
if vcpkg_found
  # Set up pkg-config and cmake paths for vcpkg
  vcpkg_include_dir = vcpkg_installed_dir / 'include'
  vcpkg_lib_dir = vcpkg_installed_dir / 'lib'
  vcpkg_bin_dir = vcpkg_installed_dir / 'bin'
  vcpkg_pkgconfig_dir = vcpkg_lib_dir / 'pkgconfig'
  vcpkg_share_pkgconfig_dir = vcpkg_installed_dir / 'share' / 'pkgconfig'
  # Note: Debug libraries not used since VCPKG_BUILD_TYPE=release in CI
  
  message('vcpkg pkg-config directories:')
  message('  - ' + vcpkg_pkgconfig_dir)
  message('  - ' + vcpkg_share_pkgconfig_dir)
  
  # Add vcpkg include and lib paths as compiler/linker flags
  # This ensures that even if pkg-config fails, the compiler can find headers and libs
  add_project_arguments(
    '-I' + vcpkg_include_dir,
    language: 'c'
  )
  
  # Add library search path (release-only since VCPKG_BUILD_TYPE=release)
  add_project_link_arguments(
    '-L' + vcpkg_lib_dir,
    language: 'c'
  )
  
  # Store vcpkg dirs for dependency lookups
  vcpkg_cmake_prefix_path = vcpkg_installed_dir
  
else
  message('vcpkg dependencies not available, using system dependencies')
  vcpkg_found = false
endif

vcpkg_library_dirs = []
if vcpkg_found
  vcpkg_library_dirs = [vcpkg_lib_dir]
endif

# Configuration data
conf_data = configuration_data()
conf_data.set('HAVE_CONFIG_H', 1)
conf_data.set_quoted('AUBIO_PREFIX', get_option('prefix'))
conf_data.set_quoted('PACKAGE', 'aubio')

# Build type
if get_option('buildtype') == 'debug'
  conf_data.set('DEBUG', 1)
else
  conf_data.set('NDEBUG', 1)
endif

# Check for standard headers
check_headers = [
  'stdlib.h',
  'stdio.h',
  'math.h',
  'string.h',
  'errno.h',
  'limits.h',
  'stdarg.h',
  'getopt.h',
  'unistd.h',
]

foreach h : check_headers
  if cc.has_header(h)
    conf_data.set('HAVE_' + h.to_upper().underscorify(), 1)
  endif
endforeach

# Platform-specific settings
if host_system == 'windows'
  conf_data.set('HAVE_WIN_HACKS', 1)
endif

# Check for math library
math_dep = cc.find_library('m', required: false)

# Double precision mode
enable_double = get_option('double')
if enable_double
  conf_data.set('HAVE_AUBIO_DOUBLE', 1)
  message('Building in double precision mode')
else
  message('Building in single precision mode')
endif

# Complex number support
if get_option('complex')
  if cc.has_header('complex.h')
    conf_data.set('HAVE_COMPLEX_H', 1)
  endif
endif

# Check for C99 __VA_ARGS__ macros
va_args_code = '''
#include <stdio.h>
#define AUBIO_ERR(...) fprintf(stderr, __VA_ARGS__)
int main(void) { return 0; }
'''
if cc.compiles(va_args_code, name: 'C99 __VA_ARGS__ macro')
  conf_data.set('HAVE_C99_VARARGS_MACROS', 1)
endif

# Memcpy hacks
if get_option('memcpy')
  conf_data.set('HAVE_MEMCPY_HACKS', 1)
endif

# Wavread/wavwrite support
if get_option('wavread')
  conf_data.set('HAVE_WAVREAD', 1)
endif
if get_option('wavwrite')
  conf_data.set('HAVE_WAVWRITE', 1)
endif

# Dependencies
dependencies = [math_dep]

# FFT implementation
fftw3_dep = dependency('', required: false)
fftw3f_dep = dependency('', required: false)

if get_option('fftw3f').enabled()
  fftw3f_dep = dependency('fftw3f', version: '>=3.0.0', required: get_option('fftw3f'))
  if fftw3f_dep.found()
    conf_data.set('HAVE_FFTW3F', 1)
    conf_data.set('HAVE_FFTW3', 1)
    dependencies += fftw3f_dep
    message('FFT implementation: fftw3f')
  endif
elif get_option('fftw3').enabled()
  fftw3_dep = dependency('fftw3', version: '>=3.0.0', required: get_option('fftw3'))
  if fftw3_dep.found()
    conf_data.set('HAVE_FFTW3', 1)
    dependencies += fftw3_dep
    message('FFT implementation: fftw3')
  endif
endif

# Intel IPP
if get_option('intelipp').enabled()
  ipp_headers = ['ippcore.h', 'ippvm.h', 'ipps.h']
  has_ipp_headers = true
  foreach h : ipp_headers
    if not cc.has_header(h)
      has_ipp_headers = false
    endif
  endforeach
  
  if has_ipp_headers
    ipp_dep = cc.find_library('ippcore', required: false)
    ippvm_dep = cc.find_library('ippvm', required: false)
    ipps_dep = cc.find_library('ipps', required: false)
    
    if ipp_dep.found() and ippvm_dep.found() and ipps_dep.found()
      conf_data.set('HAVE_INTEL_IPP', 1)
      dependencies += [ipp_dep, ippvm_dep, ipps_dep]
      message('Intel IPP support: enabled')
      
      if cc.get_id() == 'msvc'
        conf_data.set('_IPP_SEQUENTIAL_STATIC', 1)
      endif
    endif
  endif
endif

# Accelerate framework (macOS)
accelerate_dep = dependency('', required: false)
if host_system == 'darwin' and get_option('accelerate').auto()
  accelerate_dep = dependency('Accelerate', required: false)
  if accelerate_dep.found()
    conf_data.set('HAVE_ACCELERATE', 1)
    dependencies += accelerate_dep
    message('Accelerate framework: enabled')
  endif
endif

# Apple Audio (macOS/iOS)
apple_audio_deps = []
if host_system == 'darwin' and get_option('apple-audio').auto()
  corefoundation_dep = dependency('CoreFoundation', required: false)
  audiotoolbox_dep = dependency('AudioToolbox', required: false)
  
  if corefoundation_dep.found() and audiotoolbox_dep.found()
    conf_data.set('HAVE_SOURCE_APPLE_AUDIO', 1)
    conf_data.set('HAVE_SINK_APPLE_AUDIO', 1)
    apple_audio_deps = [corefoundation_dep, audiotoolbox_dep]
    dependencies += apple_audio_deps
    message('Apple Audio support: enabled')
  endif
endif

# libsndfile, jack, flac share the same detection flow
single_feature_specs = [
  {
    'option': 'sndfile',
    'pkg': 'sndfile',
    'version': '>=1.0.4',
    'fallback_libs': ['sndfile'],
    'define': 'HAVE_SNDFILE',
    'auto_message': 'libsndfile not found, disabling support automatically',
    'error_message': 'libsndfile support was requested but the dependency could not be found',
  },
  {
    'option': 'jack',
    'pkg': 'jack',
    'fallback_libs': ['jack'],
    'define': 'HAVE_JACK',
    'auto_message': 'JACK not found, disabling support automatically',
    'error_message': 'JACK support was requested but the dependency could not be found',
  },
  {
    'option': 'flac',
    'pkg': 'flac',
    'fallback_libs': ['FLAC'],
    'define': 'HAVE_FLAC',
    'auto_message': 'FLAC not found, disabling support automatically',
    'error_message': 'FLAC support was requested but the dependency could not be found',
  },
]

# On Linux, when building shared library with static dependencies,
# we must explicitly link transitive dependencies of libsndfile
# (see https://stackoverflow.com/questions/8140494)
if host_system == 'linux'
  linux_transitive_specs = [
    {'pkg': 'opus', 'fallback_libs': ['opus']},
    {'pkg': 'mp3lame', 'fallback_libs': ['mp3lame']},
    {'pkg': 'mpg123', 'fallback_libs': ['mpg123']},
    {'pkg': 'vorbis', 'fallback_libs': ['vorbis']},
    {'pkg': 'vorbisenc', 'fallback_libs': ['vorbisenc']},
    {'pkg': 'ogg', 'fallback_libs': ['ogg']},
  ]
  
  foreach spec : linux_transitive_specs
    dep = dependency(spec['pkg'], required: false, disabler: true)
    if not dep.found() and vcpkg_found and spec.has_key('fallback_libs')
      foreach lib_name : spec['fallback_libs']
        dep = cc.find_library(lib_name, dirs: vcpkg_library_dirs, required: false)
        if dep.found()
          break
        endif
      endforeach
    endif
    if dep.found()
      dependencies += dep
      message('Found transitive dependency: @0@'.format(spec['pkg']))
    endif
  endforeach
endif

foreach spec : single_feature_specs
  opt = get_option(spec['option'])
  if opt.disabled()
    continue
  endif

  has_version = spec.has_key('version')
  if has_version
    dep = dependency(spec['pkg'], version: spec['version'], required: false, disabler: true)
  else
    dep = dependency(spec['pkg'], required: false, disabler: true)
  endif

  if not dep.found() and vcpkg_found and spec.has_key('fallback_libs')
    foreach lib_name : spec['fallback_libs']
      dep = cc.find_library(lib_name, dirs: vcpkg_library_dirs, required: opt.enabled())
      if dep.found()
        break
      endif
    endforeach
  endif

  if dep.found()
    conf_data.set(spec['define'], 1)
    dependencies += dep
  elif opt.enabled()
    error(spec['error_message'])
  elif opt.auto()
    auto_msg = '@0@ not found, disabling support automatically'.format(spec['pkg'])
    if spec.has_key('auto_message')
      auto_msg = spec['auto_message']
    endif
    message(auto_msg)
  endif
endforeach

# libsamplerate (not compatible with double precision)
if not enable_double
  samplerate_opt = get_option('samplerate')
  if not samplerate_opt.disabled()
    samplerate_dep = dependency('samplerate', version: '>=0.0.15', required: false, disabler: true)
    if not samplerate_dep.found() and vcpkg_found
      samplerate_dep = cc.find_library('samplerate', dirs: vcpkg_library_dirs, required: samplerate_opt.enabled())
    endif
    if samplerate_dep.found()
      conf_data.set('HAVE_SAMPLERATE', 1)
      dependencies += samplerate_dep
    elif samplerate_opt.enabled()
      error('libsamplerate support was requested but the dependency could not be found')
    elif samplerate_opt.auto()
      message('libsamplerate not found, disabling support automatically')
    endif
  endif
else
  if get_option('samplerate').enabled()
    error('libsamplerate is not compatible with double precision mode')
  endif
endif

# rubberband (C++ library, requires C++ standard library)
rubberband_opt = get_option('rubberband')
if not rubberband_opt.disabled()
  rubberband_dep = dependency('rubberband', version: '>=1.3', required: false, disabler: true)
  if not rubberband_dep.found() and vcpkg_found
    rubberband_dep = cc.find_library('rubberband', dirs: vcpkg_library_dirs, required: rubberband_opt.enabled())
  endif
  if rubberband_dep.found()
    conf_data.set('HAVE_RUBBERBAND', 1)
    dependencies += rubberband_dep
    if host_system == 'darwin'
      add_project_link_arguments('-lc++', language: 'c')
      sleef_dep = dependency('sleef', required: false, disabler: true)
      if not sleef_dep.found() and vcpkg_found
        sleef_dep = cc.find_library('sleef', dirs: vcpkg_library_dirs, required: false)
      endif
      if sleef_dep.found()
        dependencies += sleef_dep
      endif
    endif
  elif rubberband_opt.enabled()
    error('rubberband support was requested but the dependency could not be found')
  elif rubberband_opt.auto()
    message('rubberband not found, disabling support automatically')
  endif
endif

# libav (avcodec, avformat, avutil, swresample)
avcodec_opt = get_option('avcodec')
if not avcodec_opt.disabled()
  libav_specs = [
    {'pkg': 'libavcodec', 'version': '>=54.35.0', 'fallback': 'avcodec', 'define': 'HAVE_AVCODEC'},
    {'pkg': 'libavformat', 'version': '>=52.3.0', 'fallback': 'avformat', 'define': 'HAVE_AVFORMAT'},
    {'pkg': 'libavutil', 'version': '>=52.3.0', 'fallback': 'avutil', 'define': 'HAVE_AVUTIL'},
    {'pkg': 'libswresample', 'version': '>=1.2.0', 'fallback': 'swresample', 'define': 'HAVE_SWRESAMPLE'},
  ]

  libav_deps = []
  libav_missing = false
  foreach spec : libav_specs
    dep = dependency(spec['pkg'], version: spec['version'], required: false, disabler: true)
    if not dep.found() and vcpkg_found
      dep = cc.find_library(spec['fallback'], dirs: vcpkg_library_dirs, required: avcodec_opt.enabled())
    endif
    if dep.found()
      libav_deps += dep
    else
      libav_missing = true
    endif
  endforeach

  if not libav_missing
    conf_data.set('HAVE_LIBAV', 1)
    foreach spec : libav_specs
      conf_data.set(spec['define'], 1)
    endforeach
    dependencies += libav_deps
  elif avcodec_opt.enabled()
    error('libavcodec support was requested but the dependencies could not be found')
  elif avcodec_opt.auto()
    message('libavcodec not found, disabling support automatically')
  endif
endif

# vorbis
vorbis_opt = get_option('vorbis')
if not vorbis_opt.disabled()
  vorbis_specs = [
    {'pkg': 'vorbisenc', 'fallback': 'vorbisenc'},
    {'pkg': 'vorbis', 'fallback': 'vorbis'},
    {'pkg': 'ogg', 'fallback': 'ogg'},
  ]
  vorbis_deps = []
  vorbis_missing = false
  foreach spec : vorbis_specs
    dep = dependency(spec['pkg'], required: false, disabler: true)
    if not dep.found() and vcpkg_found
      dep = cc.find_library(spec['fallback'], dirs: vcpkg_library_dirs, required: vorbis_opt.enabled())
    endif
    if dep.found()
      vorbis_deps += dep
    else
      vorbis_missing = true
    endif
  endforeach

  if not vorbis_missing
    conf_data.set('HAVE_VORBISENC', 1)
    dependencies += vorbis_deps
  elif vorbis_opt.enabled()
    error('Vorbis support was requested but the dependencies could not be found')
  elif vorbis_opt.auto()
    message('Vorbis not found, disabling support automatically')
  endif
endif

# BLAS
blas_dep = dependency('blas', required: get_option('blas'))
if blas_dep.found()
  conf_data.set('HAVE_BLAS', 1)
  dependencies += blas_dep
  
  # Check for BLAS headers
  blas_headers = ['cblas.h', 'atlas/cblas.h', 'openblas/cblas.h']
  foreach h : blas_headers
    if cc.has_header(h)
      break
    endif
  endforeach
endif

# Generate config.h
configure_file(
  output: 'config.h',
  configuration: conf_data,
  install: false,
)

# Add configuration include
config_inc = include_directories('.')

# Subdirectories
subdir('src')

if get_option('examples')
  subdir('examples')
endif

if get_option('tests')
  subdir('tests')
endif

# Python bindings
if get_option('python')
  subdir('python')
endif

# pkg-config file
pkg = import('pkgconfig')
pkg.generate(
  libaubio,
  name: 'aubio',
  description: 'a library for audio labelling',
  version: aubio_version,
  filebase: 'aubio',
  subdirs: 'aubio',
)

summary({
  'libsndfile': conf_data.has('HAVE_SNDFILE'),
  'libsamplerate': conf_data.has('HAVE_SAMPLERATE'),
  'rubberband': conf_data.has('HAVE_RUBBERBAND'),
  'JACK': conf_data.has('HAVE_JACK'),
  'FFmpeg/libav': conf_data.has('HAVE_LIBAV'),
  'Vorbis': conf_data.has('HAVE_VORBISENC'),
  'FLAC': conf_data.has('HAVE_FLAC'),
}, section: 'Optional dependencies')

