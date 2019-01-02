#include "aubio_priv.h"
#include "fmat.h"
#include "tensor.h"

aubio_tensor_t *new_aubio_tensor(uint_t ndim, uint_t *shape)
{
  aubio_tensor_t *c = AUBIO_NEW(aubio_tensor_t);
  uint_t items_per_row = 1;
  uint_t i;

  if ((sint_t)ndim <= 0) goto failure;
  for (i = 0; i < ndim; i++) {
    if ((sint_t)shape[i] <= 0) goto failure;
  }

  c->ndim = ndim;
  c->shape[0] = shape[0];
  for (i = 1; i < ndim; i++) {
    c->shape[i] = shape[i];
    items_per_row *= shape[i];
  }
  c->size = items_per_row * shape[0];
  c->buffer = AUBIO_ARRAY(smpl_t, c->size);
  c->data = AUBIO_ARRAY(smpl_t*, shape[0]);
  for (i = 0; i < c->shape[0]; i++) {
    c->data[i] = c->buffer + i * items_per_row;
  }

  return c;

failure:
  del_aubio_tensor(c);
  return NULL;
}

void del_aubio_tensor(aubio_tensor_t *c)
{
  if (c->data) {
    if (c->data[0]) {
      AUBIO_FREE(c->data[0]);
    }
    AUBIO_FREE(c->data);
  }
  AUBIO_FREE(c);
}

uint_t aubio_tensor_as_fvec(aubio_tensor_t *c, fvec_t *o) {
  if (c->ndim  != 1) return AUBIO_FAIL;
  if (c->shape[0] <= 0) return AUBIO_FAIL;
  o->length = c->shape[0];
  o->data = c->data[0];
  return AUBIO_OK;
}

uint_t aubio_fvec_as_tensor(fvec_t *o, aubio_tensor_t *c) {
  if (o == NULL) return AUBIO_FAIL;
  c->ndim = 1;
  c->shape[0] = o->length;
  c->data = &o->data;
  c->size = o->length;
  return AUBIO_OK;
}

uint_t aubio_tensor_as_fmat(aubio_tensor_t *c, fmat_t *o) {
  if (c->ndim  != 2) return AUBIO_FAIL;
  if (c->shape[0] <= 0) return AUBIO_FAIL;
  if (c->shape[1] <= 0) return AUBIO_FAIL;
  o->height = c->shape[0];
  o->length = c->shape[1];
  o->data = c->data;
  return AUBIO_OK;
}

uint_t aubio_fmat_as_tensor(fmat_t *o, aubio_tensor_t *c) {
  if (o == NULL) return AUBIO_FAIL;
  if (c == NULL) return AUBIO_FAIL;
  c->ndim = 2;
  c->shape[0] = o->height;
  c->shape[1] = o->length;
  c->size = o->height * o->length;
  c->data = o->data;
  return AUBIO_OK;
}

smpl_t aubio_tensor_max(aubio_tensor_t *t)
{
  uint_t i;
  smpl_t max = -1000000;
  for (i = 0; i < t->size; i++) {
    max = MAX(t->data[0][i], max);
  }
  return max;
}

const char_t *aubio_tensor_get_shape_string(aubio_tensor_t *t) {
  uint_t i;
  if (!t) return NULL;
  size_t offset = 2;
  static char_t shape_str[STRN_LENGTH];
  char_t shape_str_previous[STRN_LENGTH] = "(";
  for (i = 0; i < t->ndim; i++) {
    int len = snprintf(shape_str, STRN_LENGTH, "%s%d%s",
        shape_str_previous, t->shape[i], (i == t->ndim - 1) ? "" : ", ");
    strncpy(shape_str_previous, shape_str, len);
  }
  snprintf(shape_str, strnlen(shape_str, STRN_LENGTH - offset - 1) + offset,
      "%s)", shape_str_previous);
}
