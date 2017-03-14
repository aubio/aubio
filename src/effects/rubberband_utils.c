

#include "aubio_priv.h"

#ifdef HAVE_RUBBERBAND

#include <rubberband/rubberband-c.h>

// check rubberband is 1.8.1, warn if 1.3
#if !((RUBBERBAND_API_MAJOR_VERSION >= 2) && \
    (RUBBERBAND_API_MINOR_VERSION >= 5))
#warning RubberBandOptionDetectorSoft not available, \
 please upgrade rubberband to version 1.8.1 or higher
#define RubberBandOptionDetectorSoft 0x00000000
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

char** aubio_split_str(char_t* input, const char_t sep) {
  char_t** result = 0;
  uint_t count = 0;
  char_t* in_ptr = input;
  char_t* last_sep = 0;
  char_t delim[2]; delim[0] = sep; delim[1] = 0;

  // count number of elements
  while (*in_ptr) {
    if (sep == *in_ptr) {
      count++;
      last_sep = in_ptr;
    }
    in_ptr++;
  }
  // add space for trailing token.
  count += last_sep < (input + strlen(input) - 1);
  // add one more for terminating null string
  count++;

  result = malloc(sizeof(char*) * count);
  if (result) {
    size_t idx  = 0;
    char* params = strtok(input, delim);
    while (params) {
      // make sure we don't got in the wild
      assert(idx < count);
      *(result + idx++) = strdup(params);
      params = strtok(0, delim);
    }
    assert(idx == count - 1);
    // add null string at the end
    *(result + idx) = 0;
  }
  return result;
}

RubberBandOptions aubio_get_rubberband_opts(const char_t *mode)
{
  RubberBandOptions rboptions = RubberBandOptionProcessRealTime;

  if ( strcmp(mode,"crispness:0") == 0 ) {
    rboptions |= RubberBandOptionTransientsSmooth;
    rboptions |= RubberBandOptionWindowLong;
    rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:1") == 0 ) {
    rboptions |= RubberBandOptionDetectorSoft;
    rboptions |= RubberBandOptionTransientsSmooth;
    rboptions |= RubberBandOptionWindowLong;
    rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:2") == 0 ) {
    rboptions |= RubberBandOptionTransientsSmooth;
    rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "crispness:3") == 0 ) {
    rboptions |= RubberBandOptionTransientsSmooth;
  } else if ( strcmp(mode, "crispness:4") == 0 ) {
    // same as "default"
  } else if ( strcmp(mode, "crispness:5") == 0 ) {
    rboptions |= RubberBandOptionTransientsCrisp;
  } else if ( strcmp(mode, "crispness:6") == 0 ) {
    rboptions |= RubberBandOptionTransientsCrisp;
    rboptions |= RubberBandOptionWindowShort;
    rboptions |= RubberBandOptionPhaseIndependent;
  } else if ( strcmp(mode, "default") == 0 ) {
    // nothing to do
  } else {
    // attempt to parse a list of options, separated with ','
    char *modecopy = strndup(mode, PATH_MAX);
    char **params = aubio_split_str(modecopy, ',');
    uint_t i;
    if (!params) {
      return -1;
    }
    for (i = 0; *(params + i); i++) {
      if ( strcmp(params[i], "ProcessOffline" ) == 0 )        {
             rboptions = RubberBandOptionProcessOffline;
        AUBIO_WRN("rubberband_utils: RubberBandOptionProcessOffline is not available in aubio yet\n");
        // TODO: add wrapper to function study(smpl_t *input, uint_t write)
      }
      else if ( strcmp(params[i], "ProcessRealTime" ) == 0 )       rboptions |= RubberBandOptionProcessRealTime;
      else if ( strcmp(params[i], "StretchElastic" ) == 0 )        rboptions |= RubberBandOptionStretchElastic;
      else if ( strcmp(params[i], "StretchPrecise" ) == 0 )        rboptions |= RubberBandOptionStretchPrecise;
      else if ( strcmp(params[i], "TransientsCrisp" ) == 0 )       rboptions |= RubberBandOptionTransientsCrisp;
      else if ( strcmp(params[i], "TransientsMixed" ) == 0 )       rboptions |= RubberBandOptionTransientsMixed;
      else if ( strcmp(params[i], "TransientsSmooth" ) == 0 )      rboptions |= RubberBandOptionTransientsSmooth;
      else if ( strcmp(params[i], "DetectorCompound" ) == 0 )      rboptions |= RubberBandOptionDetectorCompound;
      else if ( strcmp(params[i], "DetectorPercussive" ) == 0 )    rboptions |= RubberBandOptionDetectorPercussive;
      else if ( strcmp(params[i], "DetectorSoft" ) == 0 )          rboptions |= RubberBandOptionDetectorSoft;
      else if ( strcmp(params[i], "PhaseLaminar" ) == 0 )          rboptions |= RubberBandOptionPhaseLaminar;
      else if ( strcmp(params[i], "PhaseIndependent" ) == 0 )      rboptions |= RubberBandOptionPhaseIndependent;
      else if ( strcmp(params[i], "ThreadingAuto" ) == 0 )         rboptions |= RubberBandOptionThreadingAuto;
      else if ( strcmp(params[i], "ThreadingNever" ) == 0 )        rboptions |= RubberBandOptionThreadingNever;
      else if ( strcmp(params[i], "ThreadingAlways" ) == 0 )       rboptions |= RubberBandOptionThreadingAlways;
      else if ( strcmp(params[i], "WindowStandard" ) == 0 )        rboptions |= RubberBandOptionWindowStandard;
      else if ( strcmp(params[i], "WindowShort" ) == 0 )           rboptions |= RubberBandOptionWindowShort;
      else if ( strcmp(params[i], "WindowLong" ) == 0 )            rboptions |= RubberBandOptionWindowLong;
      else if ( strcmp(params[i], "SmoothingOff" ) == 0 )          rboptions |= RubberBandOptionSmoothingOff;
      else if ( strcmp(params[i], "SmoothingOn" ) == 0 )           rboptions |= RubberBandOptionSmoothingOn;
      else if ( strcmp(params[i], "FormantShifted" ) == 0 )        rboptions |= RubberBandOptionFormantShifted;
      else if ( strcmp(params[i], "FormantPreserved" ) == 0 )      rboptions |= RubberBandOptionFormantPreserved;
      else if ( strcmp(params[i], "PitchHighSpeed" ) == 0 )        rboptions |= RubberBandOptionPitchHighSpeed;
      else if ( strcmp(params[i], "PitchHighQuality" ) == 0 )      rboptions |= RubberBandOptionPitchHighQuality;
      else if ( strcmp(params[i], "PitchHighConsistency" ) == 0 )  rboptions |= RubberBandOptionPitchHighConsistency;
      else if ( strcmp(params[i], "ChannelsApart" ) == 0 )         rboptions |= RubberBandOptionChannelsApart;
      else if ( strcmp(params[i], "ChannelsTogether" ) == 0 )      rboptions |= RubberBandOptionChannelsTogether;
      else {
        AUBIO_WRN("rubberband_utils: did not understand option '%s', should be one of: "
          "StretchElastic|StretchPrecise, TransientsCrisp|TransientsMixed|TransientsSmooth, "
          "DetectorCompound|DetectorPercussive|DetectorSoft, PhaseLaminar|PhaseIndependent, "
          "ThreadingAuto|ThreadingNever|ThreadingAlways, WindowStandard|WindowLong|WindowShort, "
          "SmoothingOn|SmoothingOff, FormantShifted|FormantPreserved, "
          "PitchHighSpeed|PitchHighQuality|PitchHighConsistency, ChannelsApart|ChannelsTogether\n"
          , params[i]);
      }
      free(params[i]);
    }
    free(params);
    free(modecopy);
  }
  return rboptions;
}

#endif
