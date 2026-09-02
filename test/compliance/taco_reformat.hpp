/* SPDX-FileCopyrightText: 2026 Binsparse Developers
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <taco.h>

#include <vector>

inline taco::TensorBase taco_reformat(const taco::TensorBase& input,
                                      const taco::Format& format) {
  taco::TensorBase output(input.getComponentType(), input.getDimensions(),
                          format, input.getFillValue());
  std::vector<taco::IndexVar> modes(input.getOrder());
  output(modes) = input(modes);
  output.compile();
  output.assemble();
  output.compute();
  output.pack();
  return output;
}

inline taco::Format taco_dense_format(int order) {
  return taco::Format(
      std::vector<taco::ModeFormatPack>((size_t) order, taco::Dense()));
}
