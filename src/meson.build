# Collect all source files
aubio_sources = files(
  'aubio_priv.h',
  'cvec.c',
  'fmat.c',
  'fvec.c',
  'lvec.c',
  'mathutils.c',
  'musicutils.c',
  'vecutils.c',
)

# Add subdirectory sources
aubio_sources += files(
  'effects/pitchshift_dummy.c',
  'effects/rubberband_utils.c',
  'effects/timestretch_dummy.c',
  'io/ioutils.c',
  'io/sink.c',
  'io/sink_wavwrite.c',
  'io/source.c',
  'io/source_wavread.c',
  'notes/notes.c',
  'onset/onset.c',
  'onset/peakpicker.c',
  'pitch/pitch.c',
  'pitch/pitchfcomb.c',
  'pitch/pitchmcomb.c',
  'pitch/pitchschmitt.c',
  'pitch/pitchspecacf.c',
  'pitch/pitchyin.c',
  'pitch/pitchyinfast.c',
  'pitch/pitchyinfft.c',
  'spectral/awhitening.c',
  'spectral/dct.c',
  'spectral/fft.c',
  'spectral/filterbank.c',
  'spectral/filterbank_mel.c',
  'spectral/mfcc.c',
  'spectral/phasevoc.c',
  'spectral/specdesc.c',
  'spectral/statistics.c',
  'spectral/tss.c',
  'synth/sampler.c',
  'synth/wavetable.c',
  'tempo/beattracking.c',
  'tempo/tempo.c',
  'temporal/a_weighting.c',
  'temporal/biquad.c',
  'temporal/c_weighting.c',
  'temporal/filter.c',
  'temporal/resampler.c',
  'utils/hist.c',
  'utils/log.c',
  'utils/parameter.c',
  'utils/scale.c',
)

# FFT implementation sources
# Always compile plain/ooura as fallback for all cases
aubio_sources += files(
  'spectral/dct_plain.c',
  'spectral/dct_ooura.c',
  'spectral/ooura_fft8g.c',
)

# Add optimized backend if available
if conf_data.has('HAVE_FFTW3') or conf_data.has('HAVE_FFTW3F')
  aubio_sources += files('spectral/dct_fftw.c')
elif conf_data.has('HAVE_ACCELERATE')
  aubio_sources += files('spectral/dct_accelerate.c')
elif conf_data.has('HAVE_INTEL_IPP')
  aubio_sources += files('spectral/dct_ipp.c')
endif

# Platform-specific sources
if host_system == 'darwin'
  if conf_data.has('HAVE_SOURCE_APPLE_AUDIO') or conf_data.has('HAVE_SINK_APPLE_AUDIO')
    aubio_sources += files(
      'io/source_apple_audio.c',
      'io/sink_apple_audio.c',
      'io/utils_apple_audio.c',
    )
  endif
  
  if conf_data.has('HAVE_AUDIO_UNIT')
    aubio_sources += files('io/audio_unit.c')
  endif
endif

# Optional library-specific sources
if conf_data.has('HAVE_SNDFILE')
  aubio_sources += files(
    'io/source_sndfile.c',
    'io/sink_sndfile.c',
  )
endif

if conf_data.has('HAVE_LIBAV')
  aubio_sources += files('io/source_avcodec.c')
endif

if conf_data.has('HAVE_VORBISENC')
  aubio_sources += files('io/sink_vorbis.c')
endif

if conf_data.has('HAVE_FLAC')
  aubio_sources += files('io/sink_flac.c')
endif

if conf_data.has('HAVE_RUBBERBAND')
  aubio_sources += files(
    'effects/pitchshift_rubberband.c',
    'effects/timestretch_rubberband.c',
  )
endif

# Windows-specific sources
if host_system == 'windows'
  aubio_sources += files('utils/windll.c')
endif

# Platform-specific string utilities
if host_system in ['darwin', 'linux', 'freebsd', 'netbsd', 'openbsd']
  aubio_sources += files('utils/strutils.c')
elif host_system == 'windows'
  # strutils.c is already conditionally included in the file itself
  aubio_sources += files('utils/strutils.c')
endif

# Include directories
aubio_inc = include_directories('.')

# Build library objects
aubio_lib_objects = static_library('aubio_objects',
  aubio_sources,
  include_directories: [aubio_inc, config_inc],
  dependencies: dependencies,
  c_args: ['-DHAVE_CONFIG_H'],
  pic: true,
  install: false,
)

# Determine which library types to build
build_shared = true
build_static = true

if host_system == 'windows'
  # On Windows, don't build the shared library since we can't create
  # a proper import library without explicit dllexport declarations.
  # The Python extension and examples link against the static library.
  build_shared = false
  build_static = true
endif

# Build shared library
if build_shared
  # On Windows, don't install the shared library since we can't create
  # an import library without explicit dllexport declarations.
  install_shared = host_system != 'windows'
  
  libaubio = shared_library('aubio',
    objects: aubio_lib_objects.extract_all_objects(recursive: true),
    include_directories: [aubio_inc, config_inc],
    dependencies: dependencies,
    version: lib_version,
    soversion: version_data.get('lt_cur').to_string(),
    install: install_shared,
    gnu_symbol_visibility: 'default',
  )
endif

# Build static library
if build_static
  libaubio_static = static_library('aubio',
    objects: aubio_lib_objects.extract_all_objects(recursive: true),
    include_directories: [aubio_inc, config_inc],
    dependencies: dependencies,
    install: false,  # Never install static library
  )
  
  # If we didn't build a shared library, make libaubio point to static
  if not build_shared
    libaubio = libaubio_static
  endif
endif

# Install headers
install_headers('aubio.h', subdir: 'aubio')

# Install headers from subdirectories, excluding *_priv.h files
subdir_headers = [
  'effects/pitchshift.h',
  'effects/timestretch.h',
  'io/audio_unit.h',
  'io/ioutils.h',
  'io/sink_apple_audio.h',
  'io/sink_sndfile.h',
  'io/sink_wavwrite.h',
  'io/sink.h',
  'io/source_apple_audio.h',
  'io/source_avcodec.h',
  'io/source_sndfile.h',
  'io/source_wavread.h',
  'io/source.h',
  'notes/notes.h',
  'onset/onset.h',
  'onset/peakpicker.h',
  'pitch/pitch.h',
  'pitch/pitchfcomb.h',
  'pitch/pitchmcomb.h',
  'pitch/pitchschmitt.h',
  'pitch/pitchspecacf.h',
  'pitch/pitchyin.h',
  'pitch/pitchyinfast.h',
  'pitch/pitchyinfft.h',
  'spectral/awhitening.h',
  'spectral/dct.h',
  'spectral/fft.h',
  'spectral/filterbank_mel.h',
  'spectral/filterbank.h',
  'spectral/mfcc.h',
  'spectral/phasevoc.h',
  'spectral/specdesc.h',
  'spectral/tss.h',
  'synth/sampler.h',
  'synth/wavetable.h',
  'tempo/beattracking.h',
  'tempo/tempo.h',
  'temporal/a_weighting.h',
  'temporal/biquad.h',
  'temporal/c_weighting.h',
  'temporal/filter.h',
  'temporal/resampler.h',
  'utils/hist.h',
  'utils/log.h',
  'utils/parameter.h',
  'utils/scale.h',
  'cvec.h',
  'fmat.h',
  'fvec.h',
  'lvec.h',
  'mathutils.h',
  'musicutils.h',
  'types.h',
  'vecutils.h',
]

foreach header : subdir_headers
  # Get the directory part to maintain structure
  header_dir = ''
  if header.contains('/')
    parts = header.split('/')
    if parts.length() > 1
      header_dir = parts[0]
    endif
  endif
  
  if header_dir != ''
    install_headers(header, subdir: 'aubio' / header_dir)
  else
    install_headers(header, subdir: 'aubio')
  endif
endforeach

# Declare dependency for use by other parts of the build
aubio_dep = declare_dependency(
  link_with: libaubio,
  include_directories: aubio_inc,
  dependencies: dependencies,
)

