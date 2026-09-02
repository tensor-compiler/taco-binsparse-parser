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
  bsp_array_t fill = read_or_zero_fill_value(argv[1], header);
  taco::TensorBase input = read_binsparse_as_taco(argv[1], header);
  taco::TensorBase reformatted = taco_reformat(input, taco_format_from_header(header));
  write_taco_as_binsparse(argv[2], reformatted, header,
                          header_has_fill(header) ? &fill : NULL);

  bsp_destroy_array_t(&fill);
  cJSON_Delete(header);
  return 0;
}
