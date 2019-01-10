#include "aubio_priv.h"
#include "fmat.h"
#include "ai/tensor.h"

void aubio_activation_relu(aubio_tensor_t *t)
{
  uint_t i;
  AUBIO_ASSERT(t);
  for (i = 0; i < t->size; i++) {
    t->buffer[i] = MAX(0, t->buffer[i]);
  }
}

void aubio_activation_sigmoid(aubio_tensor_t *t)
{
  uint_t i;
  AUBIO_ASSERT(t);
  for (i = 0; i < t->size; i++) {
    t->buffer[i] = 1. / (1. + EXP( - t->buffer[i] ));
  }
}
