

#include "aubio_priv.h"

#ifdef HAVE_RUBBERBAND

#include "rubberband/rubberband-c.h"

// check rubberband is 1.8.1, warn if 1.3
#if !((RUBBERBAND_API_MAJOR_VERSION >= 2) && \
    (RUBBERBAND_API_MINOR_VERSION >= 5))
#warning RubberBandOptionDetectorSoft not available, \
 please upgrade rubberband to version 1.8.1 or higher
#define RubberBandOptionDetectorSoft 0x00000000
#endif

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
    // failed parsing option string
    return -1;
  }
  // other options to include
  //p->rboptions |= RubberBandOptionWindowStandard;
  //p->rboptions |= RubberBandOptionSmoothingOff;
  //p->rboptions |= RubberBandOptionFormantShifted;
  //p->rboptions |= RubberBandOptionPitchHighConsistency;
  return rboptions;
}

#endif
