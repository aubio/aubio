/*
  Copyright (C) 2018 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"
#include "fmat.h"
#include "tensor.h"

#define STRN_LENGTH 40
#if !HAVE_AUBIO_DOUBLE
#define AUBIO_SMPL_TFMT "% 9.4f"
#else
#define AUBIO_SMPL_TFMT "% 9.4lf"
#endif /* HAVE_AUBIO_DOUBLE */

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
  if (!c || !o) return AUBIO_FAIL;
  o->length = c->size;
  o->data = c->buffer;
  return AUBIO_OK;
}

uint_t aubio_fvec_as_tensor(fvec_t *o, aubio_tensor_t *c) {
  if (!o || !c) return AUBIO_FAIL;
  c->ndim = 1;
  c->shape[0] = o->length;
  c->data = &o->data;
  c->buffer = o->data;
  c->size = o->length;
  return AUBIO_OK;
}

uint_t aubio_tensor_as_fmat(aubio_tensor_t *c, fmat_t *o) {
  if (!c || !o) return AUBIO_FAIL;
  o->height = c->shape[0];
  o->length = c->size / c->shape[0];
  o->data = c->data;
  // o was allocated on the stack, data[1] may be NULL
  AUBIO_WRN("aubio_tensor_as_fmat will not create a usable table of rows\n");
  return AUBIO_OK;
}

uint_t aubio_fmat_as_tensor(fmat_t *o, aubio_tensor_t *c) {
  if (!o || !c) return AUBIO_FAIL;
  c->ndim = 2;
  c->shape[0] = o->height;
  c->shape[1] = o->length;
  c->size = o->height * o->length;
  c->data = o->data;
  c->buffer = o->data[0];
  return AUBIO_OK;
}

uint_t aubio_tensor_get_subtensor(aubio_tensor_t *t, uint_t i,
    aubio_tensor_t *st)
{
  uint_t j;
  if (!t || !st) return AUBIO_FAIL;
  if (i >= t->shape[0]) {
    AUBIO_ERR("tensor: index %d out of range, only %d subtensors\n",
        i, t->shape[0]);
    return AUBIO_FAIL;
  }
  if(t->ndim > 1) {
    st->ndim = t->ndim - 1;
    for (j = 0; j < st->ndim; j++) {
      st->shape[j] = t->shape[j + 1];
    }
    for (j = st->ndim; j < AUBIO_TENSOR_MAXDIM; j++) {
      st->shape[j] = 0;
    }
    st->size = t->size / t->shape[0];
  } else {
    st->ndim = 1;
    st->shape[0] = 1;
    st->size = 1;
  }
  // st was allocated on the stack, row indices are lost
  st->data = NULL;
  st->buffer = &t->buffer[0] + st->size * i;
  return AUBIO_OK;
}

uint_t aubio_tensor_have_same_shape(aubio_tensor_t *a, aubio_tensor_t *b)
{
  uint_t n;
  if (!a || !b)
    return 0;
  if (a->ndim != b->ndim)
    return 0;

  for (n = 0; n < a->ndim; n++) {
    if (a->shape[n] != b->shape[n]) {
      return 0;
    }
  }
  return 1;
}

smpl_t aubio_tensor_max(aubio_tensor_t *t)
{
  uint_t i;
  smpl_t max = t->buffer[0];
  for (i = 0; i < t->size; i++) {
    max = MAX(t->buffer[i], max);
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
    // and space last if not the last one
    int add_space = (i < t->ndim - 1);
    // add coma first if this not last, or always if 1d
    int add_coma = add_space || (t->ndim == 1);
    int len = snprintf(shape_str, STRN_LENGTH, "%s%d%s%s",
        shape_str_previous, t->shape[i],
        add_coma ? "," : "", add_space ? " " : "");
    strncpy(shape_str_previous, shape_str, len);
  }
  snprintf(shape_str, strnlen(shape_str, STRN_LENGTH - offset - 1) + offset,
      "%s)", shape_str_previous);
  return shape_str;
}

static void aubio_tensor_print_subtensor(aubio_tensor_t *t, uint_t depth)
{
  uint_t i;
  AUBIO_MSG("[");
  for (i = 0; i < t->shape[0]; i ++) {
    AUBIO_MSG("%*s", i == 0 ? 0 : depth + 1, i == 0 ? "" : " ");
    if (t->ndim == 1) {
      AUBIO_MSG(AUBIO_SMPL_TFMT, t->buffer[i]);
    } else {
      aubio_tensor_t st;
      aubio_tensor_get_subtensor(t, i, &st);
      aubio_tensor_print_subtensor(&st, depth + 1); // recursive call
    }
    AUBIO_MSG("%s%s", (i < t->shape[0] - 1) ? "," : "",
        t->ndim == 1 ? " " : ((i < t->shape[0] - 1) ? "\n" : ""));
  }
  AUBIO_MSG("]");
}

void aubio_tensor_print(aubio_tensor_t *t)
{
  AUBIO_MSG("tensor of shape %s\n", aubio_tensor_get_shape_string(t));
  aubio_tensor_print_subtensor(t, 0);
  AUBIO_MSG("\n");
}

void aubio_tensor_matmul(aubio_tensor_t *a, aubio_tensor_t *b,
    aubio_tensor_t *c)
{
  AUBIO_ASSERT (a->shape[0] == c->shape[0]);
  AUBIO_ASSERT (a->shape[1] == b->shape[0]);
  AUBIO_ASSERT (b->shape[1] == c->shape[1]);
#if !defined(HAVE_BLAS)
  uint_t i, j, k;
  for (i = 0; i < c->shape[0]; i++) {
    for (j = 0; j < c->shape[1]; j++) {
      smpl_t sum = 0.;
      for (k = 0; k < a->shape[1]; k++) {
          sum += a->buffer[i * a->shape[1] + k]
            * b->buffer[k * b->shape[1] + j];
      }
      c->buffer[i * c->shape[1] + j] = sum;
    }
  }
#else
  aubio_cblas__gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->shape[0],
      b->size/b->shape[0], b->shape[0], 1.F, a->buffer, a->size/a->shape[0],
      b->buffer,
      b->size/b->shape[0], 0.F, c->buffer, b->size/b->shape[0]);
#endif
}

void aubio_tensor_copy(aubio_tensor_t *s, aubio_tensor_t *t)
{
  if (!aubio_tensor_have_same_shape(s, t)) {
    AUBIO_ERR("tensor: not copying source tensor %s",
        aubio_tensor_get_shape_string(s));
    AUBIO_ERR(" to destination tensor %s",
        aubio_tensor_get_shape_string(t));
    return;
  }
  AUBIO_MEMCPY(t->buffer, s->buffer, s->size);
}
