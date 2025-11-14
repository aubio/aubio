# Tests build file

# Generate test sound file
test_sound_target = '44100Hz_44100f_sine441_stereo.wav'

python3 = find_program('python3', 'python')

test_sound_gen = custom_target('generate_test_sound',
  output: test_sound_target,
  command: [
    python3,
    files('create_tests_source.py'),
    '@OUTPUT@'
  ],
)

# Get the path to the generated test sound
test_sound_path = meson.current_build_dir() / test_sound_target

# Include directories
tests_inc = include_directories('.', '../src')

# List all test source files explicitly
test_sources = files(
  # Base tests
  'src/test-cvec.c',
  'src/test-fmat.c',
  'src/test-fvec.c',
  'src/test-lvec.c',
  'src/test-mathutils.c',
  'src/test-mathutils-window.c',
  'src/test-vecutils.c',
  # Effects tests
  'src/effects/test-pitchshift.c',
  'src/effects/test-timestretch.c',
  # I/O tests
  'src/io/test-sink.c',
  'src/io/test-sink_wavwrite.c',
  'src/io/test-source.c',
  'src/io/test-source_wavread.c',
  # Notes tests
  'src/notes/test-notes.c',
  # Onset tests
  'src/onset/test-onset.c',
  'src/onset/test-peakpicker.c',
  # Pitch tests
  'src/pitch/test-pitch.c',
  'src/pitch/test-pitchfcomb.c',
  'src/pitch/test-pitchmcomb.c',
  'src/pitch/test-pitchschmitt.c',
  'src/pitch/test-pitchspecacf.c',
  'src/pitch/test-pitchyin.c',
  'src/pitch/test-pitchyinfft.c',
  # Spectral tests
  'src/spectral/test-awhitening.c',
  'src/spectral/test-dct.c',
  'src/spectral/test-fft.c',
  'src/spectral/test-filterbank.c',
  'src/spectral/test-filterbank_mel.c',
  'src/spectral/test-mfcc.c',
  'src/spectral/test-phasevoc.c',
  'src/spectral/test-specdesc.c',
  'src/spectral/test-tss.c',
  # Synth tests
  'src/synth/test-sampler.c',
  'src/synth/test-wavetable.c',
  # Tempo tests
  'src/tempo/test-beattracking.c',
  'src/tempo/test-tempo.c',
  # Temporal tests
  'src/temporal/test-a_weighting.c',
  'src/temporal/test-biquad.c',
  'src/temporal/test-c_weighting.c',
  'src/temporal/test-filter.c',
  'src/temporal/test-resampler.c',
  # Utils tests
  'src/utils/test-hist.c',
  'src/utils/test-log.c',
  'src/utils/test-parameter.c',
  'src/utils/test-scale.c',
)

# Optional tests based on enabled features
if conf_data.has('HAVE_SNDFILE')
  test_sources += files(
    'src/io/test-sink_sndfile.c',
    'src/io/test-source_sndfile.c',
  )
endif

if conf_data.has('HAVE_LIBAV')
  test_sources += files('src/io/test-source_avcodec.c')
endif

if conf_data.has('HAVE_VORBISENC')
  test_sources += files('src/io/test-sink_vorbis.c')
endif

if conf_data.has('HAVE_FLAC')
  test_sources += files('src/io/test-sink_flac.c')
endif

if host_system == 'darwin' and conf_data.has('HAVE_SOURCE_APPLE_AUDIO')
  test_sources += files(
    'src/io/test-sink_apple_audio.c',
    'src/io/test-source_apple_audio.c',
  )
endif

# Build and register each test
foreach test_src : test_sources
  # Extract test name from source file object
  # Get just the file name without path
  test_src_str = '@0@'.format(test_src)
  if test_src_str.contains('/')
    # Handle forward slash path separator
    parts = test_src_str.split('/')
    test_filename = parts[-1]
  elif test_src_str.contains('\\')
    # Handle backslash path separator  
    parts = test_src_str.split('\\')
    test_filename = parts[-1]
  else
    test_filename = test_src_str
  endif
  
  test_name = test_filename.replace('.c', '').replace('test-', '')
  test_exe_name = test_filename.replace('.c', '')
  
  test_exe = executable(test_exe_name,
    test_src,
    include_directories: [tests_inc, config_inc],
    dependencies: aubio_dep,
    c_args: [
      '-DHAVE_CONFIG_H=1',
      '-DAUBIO_UNSTABLE_API=1',
      '-DAUBIO_TESTS_SOURCE=' + test_sound_path,
    ],
    install: false,
  )
  
  # Test depends on the generated sound file
  test(test_name, test_exe, depends: test_sound_gen)
endforeach
