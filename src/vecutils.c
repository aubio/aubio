#include "config.h"
#include "aubio_priv.h"
#include "types.h"
#include "fvec.h"
#include "cvec.h"
#include "vecutils.h"

#define AUBIO_OP(OPNAME, OP, TYPE, OBJ) \
void TYPE ## _ ## OPNAME (TYPE ## _t *o) \
{ \
  uint_t j; \
  for (j = 0; j < o->length; j++) { \
    o->OBJ[j] = OP (o->OBJ[j]); \
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
AUBIO_OP_C_AND_F(log10, SAFE_LOG10)
AUBIO_OP_C_AND_F(log, SAFE_LOG)
AUBIO_OP_C_AND_F(floor, FLOOR)
AUBIO_OP_C_AND_F(ceil, CEIL)
AUBIO_OP_C_AND_F(round, ROUND)

//AUBIO_OP_C_AND_F(pow, POW)
void fvec_pow (fvec_t *s, smpl_t power)
{
  uint_t j;
  for (j = 0; j < s->length; j++) {
    s->data[j] = POW(s->data[j], power);
  }
}

void cvec_pow (cvec_t *s, smpl_t power)
{
  uint_t j;
  for (j = 0; j < s->length; j++) {
    s->norm[j] = POW(s->norm[j], power);
  }
}

