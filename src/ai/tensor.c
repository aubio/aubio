#include "aubio_priv.h"
#include "fmat.h"
#include "tensor.h"

aubio_tensor_t *new_aubio_tensor(uint_t n_dims, uint_t *dims)
{
  aubio_tensor_t *c = AUBIO_NEW(aubio_tensor_t);
  uint_t i;

  if ((sint_t)n_dims <= 0) goto failure;
  for (i = 0; i < n_dims; i++) {
    if ((sint_t)dims[i] <= 0) goto failure;
  }

  c->n_dims = n_dims;
  c->items_per_row = 1;
  //c->dims = AUBIO_ARRAY(uint_t, n_dims);
  c->dims[0] = dims[0];
  for (i = 1; i < n_dims; i++) {
    c->dims[i] = dims[i];
    c->items_per_row *= dims[i];
  }
  c->n_items = c->items_per_row * dims[0];
  c->data = AUBIO_ARRAY(smpl_t*, dims[0]);
  c->data[0] = AUBIO_ARRAY(smpl_t, c->n_items);
  for (i = 1; i < c->dims[0]; i++) {
    c->data[i] = c->data[0] + i * c->items_per_row;
  }

  return c;

failure:
  del_aubio_tensor(c);
  return NULL;
}

void del_aubio_tensor(aubio_tensor_t *c)
{
  AUBIO_ASSERT(c);
  if (c->data) {
    if (c->data[0]) {
      AUBIO_FREE(c->data[0]);
    }
    AUBIO_FREE(c->data);
  }
  //if (c->dims)
  //  AUBIO_FREE(c->dims);
  AUBIO_FREE(c);
}

uint_t aubio_tensor_as_fvec(aubio_tensor_t *c, fvec_t *o) {
  if (c->n_dims  != 1) return AUBIO_FAIL;
  if (c->dims[0] <= 0) return AUBIO_FAIL;
  o->length = c->dims[0];
  o->data = c->data[0];
  return AUBIO_OK;
}

uint_t aubio_fvec_as_tensor(fvec_t *o, aubio_tensor_t *c) {
  if (o == NULL) return AUBIO_FAIL;
  c->n_dims = 1;
  c->dims[0] = o->length;
  c->data = &o->data;
  return AUBIO_OK;
}

uint_t aubio_tensor_as_fmat(aubio_tensor_t *c, fmat_t *o) {
  if (c->n_dims  != 2) return AUBIO_FAIL;
  if (c->dims[0] <= 0) return AUBIO_FAIL;
  if (c->dims[1] <= 0) return AUBIO_FAIL;
  o->height = c->dims[0];
  o->length = c->dims[1];
  o->data = c->data;
  return AUBIO_OK;
}

uint_t aubio_fmat_as_tensor(fmat_t *o, aubio_tensor_t *c) {
  if (o == NULL) return AUBIO_FAIL;
  if (c == NULL) return AUBIO_FAIL;
  c->n_dims = 2;
  c->dims[0] = o->height;
  c->dims[1] = o->length;
  c->data = o->data;
  return AUBIO_OK;
}

smpl_t aubio_tensor_max(aubio_tensor_t *t)
{
  uint_t i;
  smpl_t max = -1000000;
  for (i = 0; i < t->n_items; i++) {
    max = MAX(t->data[0][i], max);
  }
  return max;
}
