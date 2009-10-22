#include "aubio.h"
#include "aubiocpp.h"

namespace aubio {

  fvec::fvec(uint_t length, uint_t channels) {
    self = new_fvec(length, channels);
  }

  fvec::~fvec() {
    del_fvec(self);
  }

  smpl_t* fvec::operator[]( uint_t channel ) {
    return fvec_get_channel(self, channel);
  }

  cvec::cvec(uint_t length, uint_t channels) {
    self = new_cvec(length, channels);
    norm = cvec_get_norm(self);
    phas = cvec_get_phas(self);
  }

  cvec::~cvec() {
    del_cvec(self);
  }

}
