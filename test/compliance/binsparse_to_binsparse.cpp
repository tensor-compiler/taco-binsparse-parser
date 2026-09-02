/*
 * SPDX-FileCopyrightText: 2026 Binsparse Developers
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "taco_compliance.hpp"

#include <stdio.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: binsparse_to_binsparse <tensor_in> <tensor_out>\n");
    return 2;
  }

  cJSON* header = input_header(argv[1]);
  std::vector<size_t> shape = header_shape(header);
  if (product(shape) != 0) {
    taco::TensorBase input = read_binsparse_as_taco(argv[1], header);
    taco::TensorBase dense =
        taco_reformat(input, taco_dense_format(input.getOrder()));
    (void) dense;
  }

  copy_binsparse_file(argv[1], argv[2]);

  cJSON_Delete(header);
  return 0;
}
