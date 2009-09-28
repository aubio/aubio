#include "config.h"
#include "types.h"
#include "fvec.h"
#include "cvec.h"
#include "aubio_priv.h"
#include "vecutils.h"

#define AUBIO_OP(OPNAME, OP, TYPE, OBJ) \
void TYPE ## _ ## OPNAME (TYPE ## _t *o) \
{ \
  uint_t i,j; \
  for (i = 0; i < o->channels; i++) { \
    for (j = 0; j < o->length; j++) { \
      o->OBJ[i][j] = OP (o->OBJ[i][j]); \
    } \
  } \
}

#define AUBIO_OP_C_AND_F(OPNAME, OP) \
  AUBIO_OP(OPNAME, OP, fvec, data) \
  AUBIO_OP(OPNAME, OP, cvec, norm)

AUBIO_OP_C_AND_F(exp, EXP)
AUBIO_OP_C_AND_F(cos, COS)
AUBIO_OP_C_AND_F(sin, SIN)
AUBIO_OP_C_AND_F(abs, ABS)
AUBIO_OP_C_AND_F(sqrt, SQRT)
AUBIO_OP_C_AND_F(log10, SAFELOG10)
AUBIO_OP_C_AND_F(log, SAFELOG)
AUBIO_OP_C_AND_F(floor, FLOOR)
AUBIO_OP_C_AND_F(ceil, CEIL)
AUBIO_OP_C_AND_F(round, ROUND)

//AUBIO_OP_C_AND_F(pow, POW)
void fvec_pow (fvec_t *s, smpl_t power)
{
  uint_t i,j;
  for (i = 0; i < s->channels; i++) {
    for (j = 0; j < s->length; j++) {
      s->data[i][j] = POW(s->data[i][j], power);
    }
  }
}

