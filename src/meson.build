compiler = meson.get_compiler('c')

conf_data = configuration_data()

c_args = ['-DHAVE_CONFIG_H=1']

if compiler.has_header('stdlib.h')
  conf_data.set ('HAVE_STDLIB_H', 1)
endif
if compiler.has_header('stdio.h')
  conf_data.set ('HAVE_STDIO_H', 1)
endif
# TODO: add complex
# TODO: add fftw3
if compiler.has_header('math.h')
  conf_data.set ('HAVE_MATH_H', 1)
endif
if compiler.has_header('string.h')
  conf_data.set ('HAVE_STRING_H', 1)
endif
if compiler.has_header('errno.h')
  conf_data.set ('HAVE_ERRNO_H', 1)
endif
if compiler.has_header('limits.h')
  conf_data.set ('HAVE_STDARG_H', 1)
endif
if compiler.has_header('stdarg.h')
  conf_data.set ('HAVE_STDARG_H', 1)
endif
if compiler.has_header('getopt.h', required : false)
  conf_data.set ('HAVE_GETOPT_H', 1)
endif

dependencies = []

libm = compiler.find_library('m', required : false)
if libm.found()
  dependencies += [libm]
endif

#if compiler.has_header('complex.h')
#  conf_data.set('HAVE_COMPLEX_H', true)
#endif

sndfile = dependency('sndfile', version : '>= 1.0.4', required : false)
if sndfile.found()
  dependencies += [sndfile]
  conf_data.set ('HAVE_SNDFILE', 1)
endif

samplerate = dependency('samplerate', version : '>= 0.0.15', required : false)
if samplerate.found()
  dependencies += [samplerate]
  conf_data.set ('HAVE_SAMPLERATE', 1)
endif

libavcodec = dependency('libavcodec', version : '>= 54.35.0', required : false)
libavformat = dependency('libavformat', version : '>= 52.3.0', required : false)
libavutil = dependency('libavutil', version : '>= 52.3.0', required : false)
libswresample = dependency('libswresample', version : '>= 1.2.0', required : false)
if libavcodec.found() and libavformat.found() and libavutil.found() and libswresample.found()
  dependencies += [libavcodec, libavformat, libavutil, libswresample]
  conf_data.set ('HAVE_LIBAV', 1)
endif

ogg = dependency('ogg', required : false)
vorbis = dependency('vorbis', required : false)
vorbisenc = dependency('vorbisenc', required : false)
if ogg.found() and vorbis.found() and vorbisenc.found()
  dependencies += [ogg, vorbis, vorbisenc]
  conf_data.set ('HAVE_VORBISENC', 1)
endif

flac = dependency('flac', required : false)
if flac.found()
  dependencies += [flac]
  conf_data.set ('HAVE_FLAC', 1)
endif

# todo add optiono
conf_data.set ('HAVE_WAVREAD', 1)
conf_data.set ('HAVE_WAVWRITE', 1)

rubberband = dependency('rubberband', required : false)
if rubberband.found()
  dependencies += [rubberband]
  conf_data.set ('HAVE_RUBBERBAND', 1)
endif

audiotoolbox = dependency('appleframeworks', modules : ['CoreFoundation', 'AudioToolbox'], required : false)
if audiotoolbox.found()
  dependencies += [audiotoolbox]
  conf_data.set ('HAVE_SOURCE_APPLE_AUDIO', 1)
  conf_data.set ('HAVE_SINK_APPLE_AUDIO', 1)
endif

accelerate = dependency('appleframeworks', modules : 'Accelerate', required : false)
if accelerate.found()
  dependencies += [accelerate]
  conf_data.set ('HAVE_ACCELERATE', 1)
endif

configure_file(#input : 'config.h.in',
               output : 'config.h',
               configuration : conf_data)

sources = [
    'cvec.c',
    'fmat.c',
    'fvec.c',
    'lvec.c',
    'mathutils.c',
    'musicutils.c',
    'vecutils.c',
    'io/audio_unit.c',
    'io/ioutils.c',
    'io/sink.c',
    'io/sink_apple_audio.c',
    'io/sink_flac.c',
    'io/sink_sndfile.c',
    'io/sink_vorbis.c',
    'io/sink_wavwrite.c',
    'io/source.c',
    'io/source_apple_audio.c',
    'io/source_avcodec.c',
    'io/source_sndfile.c',
    'io/source_wavread.c',
    'io/utils_apple_audio.c',
    'effects/pitchshift_dummy.c',
    'effects/pitchshift_rubberband.c',
    'effects/rubberband_utils.c',
    'effects/timestretch_dummy.c',
    'effects/timestretch_rubberband.c',
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
    'notes/notes.c',
    'spectral/awhitening.c',
    'spectral/dct.c',
    'spectral/dct_accelerate.c',
    'spectral/dct_fftw.c',
    'spectral/dct_ipp.c',
    'spectral/dct_ooura.c',
    'spectral/dct_plain.c',
    'spectral/fft.c',
    'spectral/filterbank.c',
    'spectral/filterbank_mel.c',
    'spectral/mfcc.c',
    'spectral/ooura_fft8g.c',
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
    'utils/strutils.c',
    'utils/windll.c',
]

aubio = both_libraries('aubio', sources, c_args : c_args, dependencies : dependencies,
    soversion: '5.4.8' )
