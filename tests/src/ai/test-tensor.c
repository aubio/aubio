#include "aubio.h"
#include "utils_tests.h"

#include "ai/tensor.h"

int test_1d(void)
{
  uint_t dims[1] = {11};
  aubio_tensor_t *c = new_aubio_tensor(1, dims);
  fvec_t a;
  aubio_tensor_t cp;

  assert(c);

  c->data[1][0] = 1.;
  c->data[0][9] = 1.;
  c->data[1][1] = 1.;

  aubio_tensor_print(c);

  PRINT_MSG(" created from fvec_t \n");
  // view tensor as fvec
  assert (aubio_tensor_as_fvec(c, &a) == 0);
  fvec_print(&a);
  // view fvec as tensor
  assert (aubio_fvec_as_tensor(&a, &cp) == 0);
  aubio_tensor_print(&cp);

  // wrong input
  assert (aubio_tensor_as_fvec(NULL, &a) != 0);
  assert (aubio_tensor_as_fvec(c, NULL) != 0);

  assert (aubio_fvec_as_tensor(NULL, &cp) != 0);
  assert (aubio_fvec_as_tensor(&a, NULL) != 0);

  del_aubio_tensor(c);
  return 0;
}

int test_2d(void)
{
  uint_t dims[2] = {3, 2};
  aubio_tensor_t *c = new_aubio_tensor(2, dims);
  fmat_t b;

  assert (c);
  c->data[0][1] = 2.;
  c->data[1][0] = 1.;

  aubio_tensor_print(c);
  assert (aubio_tensor_as_fmat(c, &b) == 0);
  fmat_print(&b);

  PRINT_MSG(" created from fmat_t\n");
  fmat_t *m = new_fmat(dims[0], dims[1]);
  aubio_tensor_t cp;

  fmat_ones(m);
  m->data[0][1] = 0;
  fmat_print(m);
  assert (aubio_fmat_as_tensor(m, &cp) == 0);
  aubio_tensor_print(&cp);

  // view tensor as fvec
  fvec_t vp;
  assert (aubio_tensor_as_fvec(&cp, &vp) == 0);
  fvec_print (&vp);

  assert (aubio_tensor_as_fmat(NULL, &b) != 0);
  assert (aubio_tensor_as_fmat(c, NULL) != 0);

  assert (aubio_fmat_as_tensor (NULL, &cp) != 0);
  assert (aubio_fmat_as_tensor (m, NULL) != 0);

  del_fmat(m);
  del_aubio_tensor(c);

  return 0;
}

int test_3d(void)
{
  uint_t dims[3] = {3, 2, 3};
  aubio_tensor_t *c = new_aubio_tensor(3, dims);
  assert (c);

  c->data[0][0 * 3 + 0] = 1;
  c->data[1][0 * 3 + 1] = 2;
  c->data[1][1 * 3 + 1] = 2;
  c->data[2][1 * 3 + 2] = 3;

  aubio_tensor_print(c);

  // view tensor as fmat
  fmat_t vm;
  assert (aubio_tensor_as_fmat(c, &vm) == 0);
  fmat_print (&vm);

  del_aubio_tensor(c);
  return 0;
}

int test_4d(void)
{
  uint_t d1 = 3, d2 = 2, d3 = 2, d4 = 4;
  uint_t dims[4] = {d1, d2, d3, d4};
  aubio_tensor_t *c = new_aubio_tensor(4, dims);

  c->data[0][0] = 1;
  c->data[1][1 * d3 * d4 + 1 * d4 + 2] = 2;
  c->data[0][2 * d2 * d3 * d4 + 1 * d3 * d4 + 1 * d4 + 3] = c->size;

  aubio_tensor_print(c);

  del_aubio_tensor(c);
  return 0;
}

int test_sizes(void)
{
  uint_t d1 = 3, d2 = 2, d3 = 2, d4 = 4;
  uint_t dims[4] = {d1, d2, d3, d4};
  aubio_tensor_t *a = new_aubio_tensor(4, dims);
  aubio_tensor_t *b = new_aubio_tensor(3, dims);

  assert (!aubio_tensor_have_same_size(a, b));

  del_aubio_tensor(b);
  dims[2] += 1;
  b = new_aubio_tensor(4, dims);
  assert (!aubio_tensor_have_same_size(a, b));
  del_aubio_tensor(b);
  dims[2] -= 1;

  dims[0] -= 1;
  dims[1] += 1;
  b = new_aubio_tensor(4, dims);
  assert (!aubio_tensor_have_same_size(a, b));
  del_aubio_tensor(b);

  dims[0] += 1;
  dims[1] -= 1;
  b = new_aubio_tensor(4, dims);
  assert (aubio_tensor_have_same_size(a, b));

  assert (!aubio_tensor_have_same_size(NULL, b));
  assert (!aubio_tensor_have_same_size(a, NULL));

  del_aubio_tensor(a);
  del_aubio_tensor(b);
  return 0;
}

int test_wrong_args(void)
{
  uint_t dims[3] = {3, 2, 1};
  assert (new_aubio_tensor(0, dims) == NULL);
  dims[1] = -1;
  assert (new_aubio_tensor(2, dims) == NULL);
  return 0;
}

int test_subtensor(void) {
  uint_t i;
  uint_t d1 = 4, d2 = 2, d3 = 2, d4 = 3;
  uint_t dims[4] = {d1, d2, d3, d4};
  uint_t ndim = 3;
  aubio_tensor_t *t = new_aubio_tensor(ndim, dims);

  assert (t != NULL);
  assert (t->data != NULL);

  for (i = 0; i < t->size; i++) {
    t->data[0][i] = (smpl_t)i;
  }

  aubio_tensor_print(t);

  aubio_tensor_t st;

  PRINT_MSG(" getting subtensor 1\n");
  assert (aubio_tensor_get_subtensor(t, 1, &st) == 0);
  assert (st.ndim = ndim - 1);
  assert (st.shape[0] == d2);
  assert (st.shape[1] == d3);
  assert (st.shape[2] == 0);
  assert (st.buffer[0] == 4);
  assert (st.buffer[3] == 7);
  assert (st.data == NULL);
  aubio_tensor_print(&st);

  PRINT_MSG(" check subtensor 3\n");
  assert (aubio_tensor_get_subtensor(t, 3, &st) == 0);
  aubio_tensor_print(&st);

  aubio_tensor_t sst;

  PRINT_MSG(" check subtensor 1 of subtensor 3\n");
  assert (aubio_tensor_get_subtensor(&st, 1, &sst) == 0);
  assert (sst.ndim = ndim - 2);
  assert (sst.shape[0] == d3);
  assert (sst.shape[1] == 0);
  assert (sst.buffer[0] == 14);
  assert (sst.buffer[1] == 15);
  aubio_tensor_print(&sst);

  // can get a single element as a tensor
  assert (aubio_tensor_get_subtensor(&sst, 1, &st) == 0);
  assert (st.buffer[0] == sst.buffer[1]);
  assert (&st.buffer[0] == &sst.buffer[1]);
  aubio_tensor_print(&st);

  // getting wrong tensors
  assert (aubio_tensor_get_subtensor(NULL, 0, &sst) != 0);
  assert (aubio_tensor_get_subtensor(&st, -2, &sst) != 0);
  assert (aubio_tensor_get_subtensor(&st, 2, &sst) != 0);
  assert (aubio_tensor_get_subtensor(&st, 0, NULL) != 0);

  del_aubio_tensor(t);
  return 0;
}

int test_subtensor_tricks (void)
{
  uint_t d1 = 4, d2 = 2, d3 = 2, d4 = 3;
  uint_t dims[4] = {d1, d2, d3, d4};
  uint_t ndim = 3;
  aubio_tensor_t *t = new_aubio_tensor(ndim, dims);
  // manually delete content
  free(t->data[0]);
  t->data[0] = NULL;

  del_aubio_tensor(t);
  return 0;
}

int test_maxtensor(void)
{
  uint_t d1 = 3, d2 = 2, d3 = 2, d4 = 4;
  uint_t dims[4] = {d1, d2, d3, d4};
  aubio_tensor_t *c = new_aubio_tensor(4, dims);

  c->data[0][0] = -200;
  c->data[1][1 * d3 * d4 + 1 * d4 + 2] = 2;
  c->data[0][2 * d2 * d3 * d4 + 1 * d3 * d4 + 1 * d4 + 3] = c->size;

  assert (aubio_tensor_max(c) == c->size);

  del_aubio_tensor(c);
  return 0;
}

int test_get_shape_string(void)
{
  uint_t d1 = 3, d2 = 2, d3 = 2, d4 = 4;
  uint_t dims[4] = {d1, d2, d3, d4};
  aubio_tensor_t *c = new_aubio_tensor(4, dims);

  c->data[0][0] = -200;
  c->data[1][1 * d3 * d4 + 1 * d4 + 2] = 2;
  c->data[0][2 * d2 * d3 * d4 + 1 * d3 * d4 + 1 * d4 + 3] = c->size;

  const char_t *shape_string = aubio_tensor_get_shape_string(c);
  PRINT_MSG("found shape string %s\n", shape_string);

  del_aubio_tensor(c);

  assert (aubio_tensor_get_shape_string(NULL) == NULL);

  return 0;
}

int main(void) {
  PRINT_MSG("testing 1d tensors\n");
  assert (test_1d() == 0);
  PRINT_MSG("testing 2d tensors\n");
  assert (test_2d() == 0);
  PRINT_MSG("testing 3d tensors\n");
  assert (test_3d() == 0);
  PRINT_MSG("testing 4d tensors\n");
  assert (test_4d() == 0);
  PRINT_MSG("testing aubio_tensor_have_same_size\n");
  assert (test_sizes() == 0);
  PRINT_MSG("testing new_aubio_tensor with wrong arguments\n");
  assert (test_wrong_args() == 0);
  PRINT_MSG("testing subtensors\n");
  assert (test_subtensor() == 0);
  PRINT_MSG("testing subtensors\n");
  assert (test_subtensor_tricks() == 0);
  PRINT_MSG("testing max\n");
  assert (test_maxtensor() == 0);
  PRINT_MSG("testing get_shape_string\n");
  assert (test_get_shape_string() == 0);
  return 0;
}
