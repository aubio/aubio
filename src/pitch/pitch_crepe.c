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

/* CREPE pitch algorithm

 References
 ----------

 CREPE: A Convolutional Representation for Pitch Estimation Jong Wook Kim,
 Justin Salamon, Peter Li, Juan Pablo Bello.  Proceedings of the IEEE
 International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
 2018. Available online at https://arxiv.org/abs/1802.06182

 Original implementation available at https://github.com/marl/crepe

*/

#include "aubio_priv.h"

#include "fmat.h"
#include "ai/tensor.h"
#include "ai/activation.h"
#include "ai/conv1d.h"
#include "ai/maxpool1d.h"
#include "ai/batchnorm.h"
#include "ai/dense.h"
#include "io/file_hdf5.h"
#include "utils/scale.h"

#define HDF5_FILE_PATH "crepe-model-tiny.h5"

// public prototypes
typedef struct _aubio_pitch_crepe_t aubio_pitch_crepe_t;
aubio_pitch_crepe_t *new_aubio_pitch_crepe(void);
void aubio_pitch_crepe_do(aubio_pitch_crepe_t *t, fvec_t *input, fvec_t *out);
void del_aubio_pitch_crepe(aubio_pitch_crepe_t *t);
smpl_t aubio_pitch_crepe_get_confidence (aubio_pitch_crepe_t * o);
uint_t aubio_pitch_crepe_set_tolerance(aubio_pitch_crepe_t * o, smpl_t
    tolerance);
smpl_t aubio_pitch_crepe_get_tolerance (aubio_pitch_crepe_t * o);

// static prototypes
static uint_t aubio_pitch_crepe_load_params(aubio_pitch_crepe_t *o);

struct _aubio_pitch_crepe_t
{
  // number of [conv, maxpool, batchnorm] groups
  uint_t n_layers;
  // layers
  aubio_conv1d_t **conv_layers;
  aubio_batchnorm_t **batchnorm_layers;
  aubio_maxpool1d_t **maxpool_layers;
  aubio_dense_t *dense_layer;
  // input/output tensors
  aubio_tensor_t *input_tensor;
  aubio_tensor_t **conv_output;
  aubio_tensor_t **batchnorm_output;
  aubio_tensor_t **maxpool_output;
  aubio_tensor_t *flattened;
  aubio_tensor_t *dense_output;

  smpl_t confidence;
  smpl_t tolerance;
  aubio_scale_t *scale;
};

aubio_pitch_crepe_t *new_aubio_pitch_crepe(void)
{
  aubio_pitch_crepe_t *o = AUBIO_NEW(aubio_pitch_crepe_t);
  aubio_tensor_t *block_input;
  // algorithm constants
  uint_t input_shape[2] = {1024, 1};
  uint_t capacity_modes[5] = {4, 8, 16, 24, 32};
  uint_t n_filters[6] = {32, 4, 4, 4, 8, 16};
  uint_t widths[6] = {512, 64, 64, 64, 64, 64};
  uint_t maxpool_stride[1] = {2};
  uint_t l0_stride[1] = {4};
  uint_t n_dense = 360;

  // local variables
  uint_t capacity_mode = 0;
  uint_t capacity = capacity_modes[capacity_mode];
  uint_t output_shape[2];
  uint_t i;

#if defined(HAVE_BLAS) && defined(HAVE_OPENBLAS_CBLAS_H)
  // workaround to prevent openblas from opening multiple threads, since
  // the overhead appears to be higher than using a single thread.
  openblas_set_num_threads(1);
#endif

  AUBIO_ASSERT (capacity_mode < 5 && (sint_t)capacity_mode >= 0);

  o->n_layers = 6;
  // create arrays of layers and tensors
  o->conv_layers = AUBIO_ARRAY(aubio_conv1d_t*, o->n_layers);
  o->conv_output = AUBIO_ARRAY(aubio_tensor_t*, o->n_layers);
  o->maxpool_layers = AUBIO_ARRAY(aubio_maxpool1d_t*, o->n_layers);
  o->maxpool_output = AUBIO_ARRAY(aubio_tensor_t*, o->n_layers);
  o->batchnorm_layers = AUBIO_ARRAY(aubio_batchnorm_t*, o->n_layers);
  o->batchnorm_output = AUBIO_ARRAY(aubio_tensor_t*, o->n_layers);

  if (!o->conv_layers || !o->conv_output
      || !o->maxpool_layers || !o->maxpool_output
      || !o->batchnorm_layers || !o->batchnorm_output)
    goto failure;

  // create layers
  for (i = 0; i < o->n_layers; i++) {
    uint_t kern_shape[1] = {widths[i]};
    // create convolutional layers
    o->conv_layers[i] = new_aubio_conv1d(n_filters[i] * capacity, kern_shape);
    if (!o->conv_layers[i]) goto failure;
    // set padding='same'
    if (aubio_conv1d_set_padding_mode(o->conv_layers[i], "same") != AUBIO_OK) {
      goto failure;
    }
    // set stride of first layer
    if ((i == 0) && (aubio_conv1d_set_stride(o->conv_layers[0],
            l0_stride) != AUBIO_OK) ) {
      goto failure;
    }

    // create batchnorm layers
    o->batchnorm_layers[i] = new_aubio_batchnorm();
    if (!o->batchnorm_layers[i]) goto failure;

    // create maxpool layers
    o->maxpool_layers[i] = new_aubio_maxpool1d(maxpool_stride);
    if (!o->maxpool_layers[i]) goto failure;
  }

  o->dense_layer = new_aubio_dense(n_dense);
  if (!o->dense_layer) goto failure;

  // create input/output tensors
  o->input_tensor = new_aubio_tensor(2, input_shape);
  if (!o->input_tensor) goto failure;
  block_input = o->input_tensor;
  for (i = 0; i < o->n_layers; i++) {
    // get shape of conv1d output and create its tensor
    if (aubio_conv1d_get_output_shape(o->conv_layers[i],
          block_input, output_shape))
      goto failure;
    o->conv_output[i] = new_aubio_tensor(2, output_shape);
    if (!o->conv_output[i]) goto failure;

    // get shape of batchnorm output and create its tensor
    if (aubio_batchnorm_get_output_shape(o->batchnorm_layers[i],
          o->conv_output[i], output_shape))
      goto failure;
    o->batchnorm_output[i] = new_aubio_tensor(2, output_shape);
    if (!o->batchnorm_output[i]) goto failure;

    // get shape of maxpool1d output and create its tensor
    if (aubio_maxpool1d_get_output_shape(o->maxpool_layers[i],
          o->batchnorm_output[i], output_shape))
      goto failure;
    o->maxpool_output[i] = new_aubio_tensor(2, output_shape);
    if (!o->maxpool_output[i]) goto failure;

    // set input for next block
    block_input = o->maxpool_output[i];
  }

  uint_t flattened_dim = o->maxpool_output[5]->shape[0];
  flattened_dim *= o->maxpool_output[5]->shape[1];
  uint_t dense_input[1] = {flattened_dim};
  o->flattened = new_aubio_tensor(1, dense_input);
  if (!o->flattened) goto failure;

  // permute and flatten
  aubio_tensor_t *permute_input = o->maxpool_output[5];
  AUBIO_DBG("permute:           (%d, %d) ->"
      " (%d, %d) (permutation=(2, 1))\n",
      permute_input->shape[0], permute_input->shape[1],
      permute_input->shape[1], permute_input->shape[0]);
  AUBIO_DBG("flatten:           (%d, %d) -> (%d)\n",
      permute_input->shape[1], permute_input->shape[0],
      o->flattened->shape[0]);

  if (aubio_dense_get_output_shape(o->dense_layer, o->flattened, output_shape))
    goto failure;
  o->dense_output = new_aubio_tensor(1, output_shape);
  if (!o->dense_output) goto failure;

  AUBIO_ASSERT(n_dense == output_shape[0]);

  if (aubio_pitch_crepe_load_params(o))
    goto failure;

  // map output units to midi note
  smpl_t start = 1997.379408437619;
  smpl_t end = 7180.;
  o->scale = new_aubio_scale(0., 359., start, start + end);
  if (!o->scale) goto failure;

  return o;

failure:
  del_aubio_pitch_crepe(o);
  return NULL;
}

void del_aubio_pitch_crepe(aubio_pitch_crepe_t *o)
{
  uint_t i;
  AUBIO_ASSERT(o);

  if (o->input_tensor) {
    del_aubio_tensor(o->input_tensor);
  }

  if (o->batchnorm_output) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->batchnorm_output[i])
        del_aubio_tensor(o->batchnorm_output[i]);
    }
    AUBIO_FREE(o->batchnorm_output);
  }

  if (o->batchnorm_layers) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->batchnorm_layers[i])
        del_aubio_batchnorm(o->batchnorm_layers[i]);
    }
    AUBIO_FREE(o->batchnorm_layers);
  }

  if (o->maxpool_output) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->maxpool_output[i])
        del_aubio_tensor(o->maxpool_output[i]);
    }
    AUBIO_FREE(o->maxpool_output);
  }

  if (o->maxpool_layers) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->maxpool_layers[i])
        del_aubio_maxpool1d(o->maxpool_layers[i]);
    }
    AUBIO_FREE(o->maxpool_layers);
  }

  if (o->conv_output) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->conv_output[i])
        del_aubio_tensor(o->conv_output[i]);
    }
    AUBIO_FREE(o->conv_output);
  }

  if (o->conv_layers) {
    for (i = 0; i < o->n_layers; i++) {
      if (o->conv_layers[i])
        del_aubio_conv1d(o->conv_layers[i]);
    }
    AUBIO_FREE(o->conv_layers);
  }

  if (o->flattened) {
    del_aubio_tensor(o->flattened);
  }

  if (o->dense_layer) {
    del_aubio_dense(o->dense_layer);
  }

  if (o->dense_output) {
    del_aubio_tensor(o->dense_output);
  }

  if (o->scale) {
    del_aubio_scale(o->scale);
  }

  AUBIO_FREE(o);
}

void aubio_pitch_crepe_do(aubio_pitch_crepe_t *o, fvec_t *input, fvec_t *out)
{
  uint_t i;
  AUBIO_ASSERT(o && input);
  // copy input to input tensor
  AUBIO_ASSERT(input->length == o->input_tensor->shape[0]);
  // normalize frame, removing mean and dividing by std
  smpl_t mean = fvec_mean(input);
  fvec_add(input, -mean);
  smpl_t std = 0.;
  for (i = 0; i < input->length; i++) {
    std += SQR(input->data[i]);
  }
  std = SQRT(std / (smpl_t)input->length);
  if (std < 1.e-7) std = 1;

  for (i = 0; i < input->length; i++) {
    o->input_tensor->data[0][i] = input->data[i] / std;
  }

  aubio_tensor_t *block_input = o->input_tensor;
  for (i = 0; i < o->n_layers; i++) {
    aubio_conv1d_do(o->conv_layers[i], block_input,
        o->conv_output[i]);
    // relu activation
    aubio_activation_relu(o->conv_output[i]);
    aubio_batchnorm_do(o->batchnorm_layers[i], o->conv_output[i],
        o->batchnorm_output[i]);
    aubio_maxpool1d_do(o->maxpool_layers[i], o->batchnorm_output[i],
        o->maxpool_output[i]);
    block_input = o->maxpool_output[i];
  }

  aubio_tensor_t *permute_input = o->maxpool_output[5];
  // perform flattening (permutation has no effect here, order unchanged)
  AUBIO_ASSERT (permute_input->size == o->flattened->size);
  for (i = 0; i < permute_input->size; i++) {
    o->flattened->data[0][i] = permute_input->data[0][i];
  }

  // compute dense layer
  aubio_dense_do(o->dense_layer, o->flattened, o->dense_output);

  // sigmoid activation
  aubio_activation_sigmoid(o->dense_output);

#if 0
  // print debug output
  for (i = 0; i < o->n_layers; i++) {
    AUBIO_DBG("pitch_crepe: conv1d[%d]    %f\n", i,
        aubio_tensor_max(o->conv_output[i]));
    AUBIO_DBG("pitch_crepe: batchnorm[%d] %f\n", i,
        aubio_tensor_max(o->batchnorm_output[i]));
    AUBIO_DBG("pitch_crepe: maxpool1d[%d] %f\n", i,
        aubio_tensor_max(o->maxpool_output[i]));
  }
  AUBIO_DBG("pitch_crepe: dense %f\n", aubio_tensor_max(o->dense_output));
#endif

  // find maximum activation
  fvec_t activations;
  aubio_tensor_as_fvec(o->dense_output, &activations);
  uint_t argmax = fvec_max_elem(&activations);
  o->confidence = activations.data[argmax];

  // skip frames with no activation at all (e.g. silence)
  // or with insufficient confidence
  if ((argmax == activations.length - 1)
      || (o->confidence < o->tolerance)) {
    out->data[0] = -100.;
    o->confidence = 0;
    return;
  }

  // perform interpolation across neighbouring outputs
  sint_t start = MAX(0, (sint_t)argmax - 4);
  uint_t end = MIN(argmax + 5, activations.length);

  smpl_t prod = 0;
  smpl_t weight = 0;
  smpl_t scaling = 0;
  for (i = start; i < end; i++) {
    scaling = (smpl_t)(i);
    prod += activations.data[i] * scaling;
    weight += activations.data[i];
  }
  out->data[0] = prod / weight;

  // map output units to midi output
  aubio_scale_do(o->scale, out);

  // convert cents to midi
  out->data[0] /= 100.;

  // final bias (f_ref = 10Hz -> 3.48 midi)
  out->data[0] += 3.486821174621582;
}

smpl_t aubio_pitch_crepe_get_confidence (aubio_pitch_crepe_t* o)
{
  return o->confidence;
}

uint_t aubio_pitch_crepe_set_tolerance(aubio_pitch_crepe_t * o,
    smpl_t tolerance)
{
  if (o->tolerance < 0 || o->tolerance > 1) return AUBIO_FAIL;
  o->tolerance = tolerance;
  return AUBIO_OK;
}

smpl_t aubio_pitch_crepe_get_tolerance (aubio_pitch_crepe_t * o)
{
  return o->tolerance;
}

uint_t aubio_pitch_crepe_load_params(aubio_pitch_crepe_t *o)
{
#ifdef HAVE_HDF5
  uint_t i;
  aubio_tensor_t *k = NULL;
  fvec_t *vec = NULL;

  AUBIO_ASSERT(o);

  aubio_file_hdf5_t *hdf5 = new_aubio_file_hdf5(HDF5_FILE_PATH);
  if (!hdf5) return AUBIO_FAIL;

  // get kernels
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d/conv%d_3/kernel:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    k = aubio_conv1d_get_kernel(o->conv_layers[i]);

    // push dimension
    k->shape[3] = k->shape[2]; k->shape[2] = k->shape[1]; k->shape[1] = 1;
    k->ndim += 1;
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_tensor(hdf5, key, k))
      return AUBIO_FAIL;
    // pop dimension
    k->shape[1] = k->shape[2]; k->shape[2] = k->shape[3]; k->shape[3] = 0;
    k->ndim -= 1;
  }

  // get bias vectors
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d/conv%d_3/bias:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    vec = aubio_conv1d_get_bias(o->conv_layers[i]);
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, vec))
      return AUBIO_FAIL;
  }

  // batchnorm
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d-BN/conv%d-BN_3/gamma:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    // get kernel matrix
    vec = aubio_batchnorm_get_gamma(o->batchnorm_layers[i]);
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, vec))
      return AUBIO_FAIL;
  }
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d-BN/conv%d-BN_3/beta:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    // get kernel matrix
    vec = aubio_batchnorm_get_beta(o->batchnorm_layers[i]);
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, vec))
      return AUBIO_FAIL;
  }
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d-BN/conv%d-BN_3/moving_mean:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    // get kernel matrix
    vec = aubio_batchnorm_get_moving_mean(o->batchnorm_layers[i]);
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, vec))
      return AUBIO_FAIL;
  }
  for (i = 0; i < o->n_layers; i++) {
    char_t *fmt_key = "/conv%d-BN/conv%d-BN_3/moving_variance:0";
    char_t key[PATH_MAX];
    snprintf(key, sizeof(key), fmt_key, i+1, i+1);
    // get kernel matrix
    vec = aubio_batchnorm_get_moving_variance(o->batchnorm_layers[i]);
    // load params from hdf5 into kernel tensor
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, vec))
      return AUBIO_FAIL;
  }

  // dense layer
  {
    char_t *key = "/classifier/classifier_3/kernel:0";
    fmat_t *d = aubio_dense_get_weights(o->dense_layer);
    if (aubio_file_hdf5_load_dataset_into_matrix(hdf5, key, d))
      return AUBIO_FAIL;

    key = "/classifier/classifier_3/bias:0";
    fvec_t *v = aubio_dense_get_bias(o->dense_layer);
    if (aubio_file_hdf5_load_dataset_into_vector(hdf5, key, v))
      return AUBIO_FAIL;
  }

  if (hdf5) {
    del_aubio_file_hdf5(hdf5);
  }

  return AUBIO_OK;
#else
  AUBIO_ASSERT(o);
  AUBIO_ERR("pitch_crepe: hdf5 support was not built in, failed loading"
      " crepe model\n");
  return AUBIO_FAIL;
#endif
}
