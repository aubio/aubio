#include "aubio.h"

namespace aubio {

  class fvec {

    private:
      fvec_t * self;

    public:
      fvec(uint_t length, uint_t channels);
      ~fvec();
      smpl_t* operator[]( uint_t channel );

  };

  class cvec {

    private:
      cvec_t * self;

    public:
      smpl_t ** norm;
      smpl_t ** phas;

      cvec(uint_t length, uint_t channels);
      ~cvec();

  };

}
