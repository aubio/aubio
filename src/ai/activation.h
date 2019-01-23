
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

#ifndef AUBIO_ACTIVATION_H
#define AUBIO_ACTIVATION_H

/** \file

  Activation functions.

*/

#ifdef __cplusplus
extern "C" {
#endif

/** Rectifier linear unit (ReLU)

  \param t input tensor on which to compute the activation function (in-place)

  Upon return, each value \f$x\f$ of the tensor \f$t\f$ will be set to
  \f$max(0, x)\f$.

*/
void aubio_activation_relu(aubio_tensor_t *t);

/** compute sigmoid activation

  \param t input tensor on which to compute the activation function (in-place)

  Upon return, each value \f$x\f$ of the tensor \f$t\f$ will be set to
  \f$\frac{1}{1 + e^{-x}}\f$.

*/
void aubio_activation_sigmoid(aubio_tensor_t *t);

#endif
