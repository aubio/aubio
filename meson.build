project('aubio', 'c',
  version: '0.4.12', # x-release-please-version
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

if aubio_version != meson.project_version()
  error('VERSION (' + aubio_version + ') and meson.build (' + meson.project_version() + ') disagree; release-please keeps them in step')
endif

message('Building aubio version: ' + aubio_version)
message('Library version: ' + lib_version)

# Compiler setup
cc = meson.get_compiler('c')
host_system = host_machine.system()

# Ensure POSIX feature macros expose prototypes like strerror_r on glibc
if host_system != 'windows'
  # 200809L, as Python.h defines it: a different value is a redefinition
  # warning in every file of the extension
  add_project_arguments('-D_POSIX_C_SOURCE=200809L', language: 'c')
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
  # Release-only static triplets, as built by the CI workflow
  if host_machine.cpu_family() == 'aarch64'
    vcpkg_triplet = 'arm64-osx-release'
  else
    vcpkg_triplet = 'x64-osx-release'
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
# smpl_t is chosen in the public header types.h, so code built against
# libaubio (its tests and examples here, and users through aubio.pc) must
# see the same HAVE_AUBIO_DOUBLE as the library, not only config.h.
aubio_public_c_args = []
if enable_double
  conf_data.set('HAVE_AUBIO_DOUBLE', 1)
  aubio_public_c_args += ['-DHAVE_AUBIO_DOUBLE=1']
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

# FFT implementation. FFTW must match the precision (fft.c refuses a
# mismatch): fftw3f for single precision, fftw3 for double. Both default to
# disabled, so ooura is used unless FFTW is asked for: ooura is power-of-two
# only, but at the sizes aubio is normally run with it was as fast or faster
# than FFTW (e.g. a 4096-point pvoc, 82us vs 85us), and FFTW_ESTIMATE plans
# are not tuned. -Dfftw3f=enabled adds non-power-of-two sizes.
fftw3_dep = dependency('', required: false)
fftw3f_dep = dependency('', required: false)

if not enable_double
  if get_option('fftw3').enabled()
    error('fftw3 is the double-precision FFTW: use -Dfftw3f=enabled, or -Ddouble=true')
  endif
  fftw3f_dep = dependency('fftw3f', version: '>=3.0.0', required: get_option('fftw3f'))
  if fftw3f_dep.found()
    conf_data.set('HAVE_FFTW3F', 1)
    conf_data.set('HAVE_FFTW3', 1)
    dependencies += fftw3f_dep
  endif
else
  if get_option('fftw3f').enabled()
    error('fftw3f is the single-precision FFTW: use -Dfftw3=enabled with -Ddouble=true')
  endif
  fftw3_dep = dependency('fftw3', version: '>=3.0.0', required: get_option('fftw3'))
  if fftw3_dep.found()
    conf_data.set('HAVE_FFTW3', 1)
    dependencies += fftw3_dep
  endif
endif

# Intel IPP
if not get_option('intelipp').disabled()
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
  if get_option('intelipp').enabled() and not conf_data.has('HAVE_INTEL_IPP')
    error('Intel IPP support was requested but its headers or libraries could not be found')
  endif
endif

# MSVC deprecates the standard C and POSIX functions (strncpy, fopen, strdup)
# in favour of its own non-portable variants (strncpy_s, _strdup, ...).
if cc.get_id() == 'msvc'
  add_project_arguments('-D_CRT_SECURE_NO_WARNINGS',
    '-D_CRT_NONSTDC_NO_DEPRECATE', language: 'c')
endif

# smpl_t math stays in smpl_t, and setup math done in double is narrowed
# once, explicitly: an implicit double -> float or float -> integer
# conversion is a mistake to report, as MSVC does by default (C4244, C4305).
add_project_arguments(cc.get_supported_arguments('-Wfloat-conversion'), language: 'c')

# Calling an Apple API newer than the deployment target, without checking
# it's there, makes the library fail to load on older macOS: an error, not a
# warning.
if host_system == 'darwin'
  add_project_arguments(cc.get_supported_arguments('-Werror=unguarded-availability-new'), language: 'c')
endif

# Accelerate framework (macOS)
accelerate_opt = get_option('accelerate').require(host_system == 'darwin',
  error_message: 'the Accelerate framework is only available on macOS')
accelerate_dep = dependency('', required: false)
if not accelerate_opt.disabled()
  accelerate_dep = dependency('Accelerate', required: accelerate_opt)
  if accelerate_dep.found()
    conf_data.set('HAVE_ACCELERATE', 1)
    dependencies += accelerate_dep
    message('Accelerate framework: enabled')
  endif
endif

# Apple Audio (macOS/iOS)
apple_audio_deps = []
apple_audio_opt = get_option('apple-audio').require(host_system == 'darwin',
  error_message: 'Apple AudioToolbox is only available on macOS')
if not apple_audio_opt.disabled()
  corefoundation_dep = dependency('CoreFoundation', required: apple_audio_opt)
  audiotoolbox_dep = dependency('AudioToolbox', required: apple_audio_opt)

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
    # only the example programs talk to JACK (examples/jackio.c, which link
    # it themselves): libaubio must not depend on libjack
    'link_library': false,
    # JACK ships a .pc file; a CMake lookup only warns about other packages'
    # CMake configs
    'method': 'pkg-config',
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
    {'pkg': 'libmpg123', 'fallback_libs': ['mpg123']},
    {'pkg': 'vorbis', 'fallback_libs': ['vorbis']},
    {'pkg': 'vorbisenc', 'fallback_libs': ['vorbisenc']},
    {'pkg': 'ogg', 'fallback_libs': ['ogg']},
  ]

  # pkg-config only: these are vcpkg's .pc files or nothing, and meson's CMake
  # fallback only adds warnings about vcpkg's CMake configs
  foreach spec : linux_transitive_specs
    dep = dependency(spec['pkg'], method: 'pkg-config', required: false, disabler: true)
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

# set by the loop below when JACK is found; used by examples/ only
jack_dep = dependency('', required: false)

foreach spec : single_feature_specs
  opt = get_option(spec['option'])
  if opt.disabled()
    continue
  endif

  method = spec.get('method', 'auto')
  if spec.has_key('version')
    dep = dependency(spec['pkg'], version: spec['version'], method: method, required: false, disabler: true)
  else
    dep = dependency(spec['pkg'], method: method, required: false, disabler: true)
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
    if spec.get('link_library', true)
      dependencies += dep
    endif
    if spec['option'] == 'jack'
      jack_dep = dep
    endif
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
    # a static rubberband needs the C++ runtime the C extension doesn't link
    if host_system == 'linux'
      add_project_link_arguments('-lstdc++', language: 'c')
    endif
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
    # A static FFmpeg's x86 assembly refers to its own constants directly
    # (R_X86_64_PC32 against ff_pw_9 ...), which a shared object only allows
    # when they bind locally. vcpkg's FFmpeg port says to link with
    # -Bsymbolic, as FFmpeg's own shared libraries do. It also keeps the
    # extension's copies of these libraries from being interposed.
    if host_system == 'linux' and vcpkg_found
      add_project_link_arguments('-Wl,-Bsymbolic', language: 'c')
    endif
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
  extra_cflags: aubio_public_c_args,
)

# What this build actually uses. Wheel builds print it too (cibuildwheel
# build-verbosity in pyproject.toml), so CI logs show each platform's
# feature set.
if conf_data.has('HAVE_FFTW3F')
  fft_backend = 'fftw3f'
elif conf_data.has('HAVE_FFTW3')
  fft_backend = 'fftw3'
elif conf_data.has('HAVE_ACCELERATE')
  fft_backend = 'Accelerate (vDSP)'
elif conf_data.has('HAVE_INTEL_IPP')
  fft_backend = 'Intel IPP'
else
  fft_backend = 'ooura (power-of-two sizes only)'
endif

summary({
  'Precision': enable_double ? 'double' : 'single',
  'Security hardening': get_option('security_hardening'),
  'Python bindings': get_option('python'),
  'C tests': get_option('tests'),
  'Examples / CLI tools': get_option('examples'),
}, section: 'Configuration')

summary({
  'FFT': fft_backend,
  'Accelerate (vector ops)': conf_data.has('HAVE_ACCELERATE'),
  'Intel IPP': conf_data.has('HAVE_INTEL_IPP'),
  'BLAS': conf_data.has('HAVE_BLAS'),
}, section: 'Math')

summary({
  'libsndfile': conf_data.has('HAVE_SNDFILE'),
  'FFmpeg/libav': conf_data.has('HAVE_LIBAV'),
  'Apple AudioToolbox': conf_data.has('HAVE_SOURCE_APPLE_AUDIO'),
  'WAV reader / writer': '@0@ / @1@'.format(conf_data.has('HAVE_WAVREAD'), conf_data.has('HAVE_WAVWRITE')),
  'Vorbis sink': conf_data.has('HAVE_VORBISENC'),
  'FLAC sink': conf_data.has('HAVE_FLAC'),
  'JACK (examples only)': conf_data.has('HAVE_JACK'),
}, section: 'Audio I/O')

summary({
  'libsamplerate': conf_data.has('HAVE_SAMPLERATE'),
  'rubberband': conf_data.has('HAVE_RUBBERBAND'),
}, section: 'Resampling and effects')
