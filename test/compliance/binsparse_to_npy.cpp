/*
 * SPDX-FileCopyrightText: 2026 Binsparse Developers
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "npy_c.h"
#include "taco_compliance.hpp"

#include <stdint.h>
#include <stdio.h>

#include <vector>

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: binsparse_to_npy <tensor_in> <tensor_out> "
                    "<pattern_out> <fill_value_out>\n");
    return 2;
  }

  cJSON* header = input_header(argv[1]);
  std::vector<size_t> shape = header_shape(header);
  bsp_type_t value_type = header_value_type(header);
  if (product(shape) == 0) {
    if (npy_c_save(argv[2], NULL, shape.empty() ? NULL : shape.data(),
                   shape.size(), value_type)) {
      die("cannot write dense tensor NPY");
    }
    if (npy_c_save(argv[3], NULL, shape.empty() ? NULL : shape.data(),
                   shape.size(), BSP_BINT8)) {
      die("cannot write pattern NPY");
    }
    bsp_array_t fill = read_or_zero_fill_value(argv[1], header);
    if (npy_c_save(argv[4], fill.data, NULL, 0, value_type)) {
      die("cannot write fill-value NPY");
    }
    bsp_destroy_array_t(&fill);
    cJSON_Delete(header);
    return 0;
  }

  taco::TensorBase input = read_binsparse_as_taco(argv[1], header);
  taco::TensorBase dense =
      taco_reformat(input, taco_dense_format(input.getOrder()));
  taco::Array dense_values = dense.getStorage().getValues();
  if (npy_c_save(argv[2], dense_values.getData(),
                  shape.empty() ? NULL : shape.data(), shape.size(),
                  value_type)) {
    die("cannot write dense tensor NPY");
  }

  std::vector<std::vector<size_t> > coordinates = stored_coordinates(input);
  std::vector<uint8_t> pattern(product(shape), 0);
  for (const std::vector<size_t>& coord : coordinates) {
    pattern[dense_offset(shape, coord.empty() ? NULL : coord.data())] = 1;
  }
  if (npy_c_save(argv[3], pattern.empty() ? NULL : pattern.data(),
                  shape.empty() ? NULL : shape.data(), shape.size(),
                  BSP_BINT8)) {
    die("cannot write pattern NPY");
  }

  bsp_array_t fill = read_or_zero_fill_value(argv[1], header);
  if (npy_c_save(argv[4], fill.data, NULL, 0, value_type)) {
    die("cannot write fill-value NPY");
  }

  bsp_destroy_array_t(&fill);
  cJSON_Delete(header);
  return 0;
}
