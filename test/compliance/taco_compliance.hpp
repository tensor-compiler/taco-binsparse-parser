/* SPDX-FileCopyrightText: 2026 Binsparse Developers
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "reformat.h"
#include "taco_reformat.hpp"

#include <binsparse/hdf5_wrapper.h>
#include <bsp_taco.hpp>
#include <cJSON/cJSON.h>
#include <hdf5.h>
#include <taco.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

inline size_t header_nnz(cJSON* header) {
  return (size_t) cJSON_GetNumberValue(
      cJSON_GetObjectItemCaseSensitive(header, "number_of_stored_values"));
}

inline std::vector<size_t> header_shape(cJSON* header) {
  cJSON* shape_json = cJSON_GetObjectItemCaseSensitive(header, "shape");
  int rank = cJSON_GetArraySize(shape_json);
  std::vector<size_t> shape((size_t) rank);
  for (int d = 0; d < rank; ++d) {
    shape[(size_t) d] =
        (size_t) cJSON_GetNumberValue(cJSON_GetArrayItem(shape_json, d));
  }
  return shape;
}

inline size_t product(const std::vector<size_t>& shape) {
  size_t result = 1;
  for (size_t dimension : shape) {
    result *= dimension;
  }
  return result;
}

inline const char* header_format(cJSON* header) {
  const char* format = cJSON_GetStringValue(
      cJSON_GetObjectItemCaseSensitive(header, "format"));
  if (format == NULL) {
    die("header has no format");
  }
  return format;
}

inline bsp_type_t header_value_type(cJSON* header) {
  cJSON* data_types = cJSON_GetObjectItemCaseSensitive(header, "data_types");
  const char* value_name = cJSON_GetStringValue(
      cJSON_GetObjectItemCaseSensitive(data_types, "values"));
  bsp_type_t type = type_from_name(value_name);
  if (type == BSP_INVALID_TYPE) {
    die("unsupported values type");
  }
  return type;
}

inline bsp_array_t read_hdf5_array(const char* path, const char* name) {
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    die("cannot open Binsparse file");
  }
  bsp_array_t array;
  if (bsp_read_array(&array, file, name) != BSP_SUCCESS) {
    H5Fclose(file);
    die("cannot read Binsparse array");
  }
  H5Fclose(file);
  if (array.data == NULL) {
    die("cannot read Binsparse array");
  }
  return array;
}

inline bsp_array_t read_or_zero_fill_value(const char* path, cJSON* header) {
  bsp_type_t type = header_value_type(header);
  if (header_has_fill(header)) {
    bsp_array_t fill = read_hdf5_array(path, "fill_value");
    fill.type = type;
    return fill;
  }
  bsp_array_t fill;
  if (bsp_construct_array_t(&fill, 1, type) != BSP_SUCCESS) {
    die("out of memory");
  }
  memset(fill.data, 0, bsp_type_size(type));
  return fill;
}

inline bool predefined_is_column_major(const char* format) {
  return !strcmp(format, "DMATC") || !strcmp(format, "CSC") ||
         !strcmp(format, "DCSC") || !strcmp(format, "COOC");
}

inline std::vector<size_t> source_transpose(cJSON* header) {
  std::vector<size_t> transpose(header_shape(header).size());
  for (size_t d = 0; d < transpose.size(); ++d) {
    transpose[d] = d;
  }

  const char* format = header_format(header);
  if (!strcmp(format, "custom")) {
    cJSON* custom = cJSON_GetObjectItemCaseSensitive(header, "custom");
    cJSON* json = cJSON_GetObjectItemCaseSensitive(custom, "transpose");
    if (json != NULL) {
      for (size_t d = 0; d < transpose.size(); ++d) {
        transpose[d] = (size_t) cJSON_GetNumberValue(
            cJSON_GetArrayItem(json, (int) d));
      }
    }
  } else if (transpose.size() == 2 && predefined_is_column_major(format)) {
    transpose[0] = 1;
    transpose[1] = 0;
  }
  return transpose;
}

inline entries_t custom_tensor_logical_entries(const char* path,
                                               cJSON* header) {
  std::vector<size_t> shape = header_shape(header);
  entries_t entries = {0};
  if (shape.empty()) {
    entries.rank = 0;
    if (header_nnz(header) != 0) {
      push_entry(&entries, NULL, 0);
    }
    return entries;
  }

  bsp_tensor_t tensor = bsp_read_tensor(path, NULL);
  tensor.is_iso = header_values_are_iso(header);
  entries = tensor_to_coo(&tensor);
  std::vector<size_t> source = source_transpose(header);
  std::vector<size_t> target(source.size());
  for (size_t d = 0; d < target.size(); ++d) {
    target[d] = d;
  }
  apply_transpose(&entries, source.data(), target.data());
  bsp_destroy_tensor_t(tensor);
  return entries;
}

inline entries_t predefined_matrix_logical_entries(const char* path,
                                                   cJSON* header) {
  bsp_matrix_t matrix;
  if (bsp_read_matrix(&matrix, path, NULL) != BSP_SUCCESS) {
    die("failed to parse input matrix");
  }
  entries_t entries = matrix_to_coo(&matrix);
  std::vector<size_t> source = source_transpose(header);
  std::vector<size_t> target(source.size());
  for (size_t d = 0; d < target.size(); ++d) {
    target[d] = d;
  }
  apply_transpose(&entries, source.data(), target.data());
  bsp_destroy_matrix_t(&matrix);
  return entries;
}

inline entries_t logical_entries_from_binsparse(const char* path,
                                                cJSON* header) {
  return !strcmp(header_format(header), "custom")
             ? custom_tensor_logical_entries(path, header)
             : predefined_matrix_logical_entries(path, header);
}

inline cJSON* coo_level_for_rank(int rank) {
  cJSON* level = cJSON_CreateObject();
  if (rank == 0) {
    cJSON_AddStringToObject(level, "level_desc", "element");
    return level;
  }
  cJSON_AddStringToObject(level, "level_desc", "sparse");
  cJSON_AddNumberToObject(level, "rank", rank);
  cJSON_AddItemToObject(level, "level", coo_level_for_rank(0));
  return level;
}

inline cJSON* int32_coo_data_types(int rank, bsp_type_t value_type) {
  cJSON* data_types = cJSON_CreateObject();
  cJSON_AddStringToObject(data_types, "values", bsp_get_type_string(value_type));
  for (int d = 0; d < rank; ++d) {
    char key[64];
    snprintf(key, sizeof(key), "indices_%d", d);
    cJSON_AddStringToObject(data_types, key, "int32");
  }
  return data_types;
}

inline bsp_tensor_t predefined_matrix_as_tensor(const char* path,
                                               cJSON* header) {
  bsp_matrix_t matrix;
  if (bsp_read_matrix(&matrix, path, NULL) != BSP_SUCCESS) {
    die("failed to parse input matrix");
  }
  matrix.is_iso = header_values_are_iso(header);
  if (matrix.is_iso) {
    die("TACO does not support iso-valued tensor storage");
  }

  std::vector<size_t> shape = header_shape(header);
  entries_t entries = matrix_to_coo(&matrix);
  std::vector<size_t> source = source_transpose(header);
  std::vector<size_t> target(source.size());
  for (size_t d = 0; d < target.size(); ++d) {
    target[d] = d;
  }
  apply_transpose(&entries, source.data(), target.data());

  bsp_tensor_t tensor = bsp_construct_default_tensor_t();
  tensor.rank = (int) shape.size();
  tensor.nnz = entries.size;
  tensor.is_iso = false;
  tensor.structure = matrix.structure;
  tensor.dims = tensor.rank ? (size_t*) malloc(shape.size() * sizeof(size_t))
                            : NULL;
  tensor.transpose =
      tensor.rank ? (size_t*) malloc(shape.size() * sizeof(size_t)) : NULL;
  for (int d = 0; d < tensor.rank; ++d) {
    tensor.dims[d] = shape[(size_t) d];
    tensor.transpose[d] = (size_t) d;
  }

  cJSON* level = coo_level_for_rank(tensor.rank);
  cJSON* data_types = int32_coo_data_types(tensor.rank, matrix.values.type);
  reformat_context_t context = {data_types};
  bsp_array_t values = values_in_entry_order(matrix.values, &entries, false);
  size_t root_ptr[2] = {0, entries.size};
  tensor.level = build_level(&context, level, &entries, tensor.dims, 0,
                             root_ptr, 1, true, values);

  cJSON_Delete(level);
  cJSON_Delete(data_types);
  free_entries(&entries);
  bsp_destroy_matrix_t(&matrix);
  return tensor;
}

inline bsp_tensor_t custom_tensor_from_binsparse(const char* path,
                                                cJSON* header) {
  std::vector<size_t> shape = header_shape(header);
  if (!shape.empty()) {
    bsp_tensor_t tensor = bsp_read_tensor(path, NULL);
    tensor.is_iso = header_values_are_iso(header);
    if (tensor.is_iso) {
      die("TACO does not support iso-valued tensor storage");
    }
    return tensor;
  }

  bsp_tensor_t tensor = bsp_construct_default_tensor_t();
  tensor.rank = 0;
  tensor.nnz = header_nnz(header);
  tensor.level = (bsp_level_t*) malloc(sizeof(bsp_level_t));
  tensor.level->kind = BSP_TENSOR_ELEMENT;
  bsp_element_t* element = (bsp_element_t*) malloc(sizeof(bsp_element_t));
  element->values = read_hdf5_array(path, "values");
  tensor.level->data = element;
  return tensor;
}

inline taco::TensorBase read_binsparse_as_taco(const char* path,
                                               cJSON* header) {
  bsp_tensor_t tensor = !strcmp(header_format(header), "custom")
                            ? custom_tensor_from_binsparse(path, header)
                            : predefined_matrix_as_tensor(path, header);
  bsp_array_t fill = read_or_zero_fill_value(path, header);
  taco::TensorBase result = bsp_taco::makeTacoTensor(tensor, &fill);
  bsp_destroy_array_t(&fill);
  bsp_destroy_tensor_t(tensor);
  return result;
}

inline void append_taco_modes(cJSON* level,
                              std::vector<taco::ModeFormatPack>* modes) {
  const char* desc = cJSON_GetStringValue(
      cJSON_GetObjectItemCaseSensitive(level, "level_desc"));
  if (desc == NULL) {
    die("invalid target level descriptor");
  }
  if (!strcmp(desc, "element")) {
    return;
  }
  int rank = (int) cJSON_GetNumberValue(
      cJSON_GetObjectItemCaseSensitive(level, "rank"));
  cJSON* child = cJSON_GetObjectItemCaseSensitive(level, "level");
  if (rank <= 0 || !cJSON_IsObject(child)) {
    die("invalid target level descriptor");
  }
  if (!strcmp(desc, "dense")) {
    for (int d = 0; d < rank; ++d) {
      modes->push_back(taco::Dense());
    }
  } else if (!strcmp(desc, "sparse")) {
    std::vector<taco::ModeFormat> pack;
    pack.push_back(
        taco::Compressed({taco::ModeFormat::ORDERED,
                          taco::ModeFormat::NOT_UNIQUE}));
    for (int d = 1; d < rank; ++d) {
      pack.push_back(taco::Singleton({taco::ModeFormat::ORDERED,
                                      taco::ModeFormat::NOT_UNIQUE}));
    }
    modes->push_back(taco::ModeFormatPack(pack));
  } else {
    die("unknown target level descriptor");
  }
  append_taco_modes(child, modes);
}

inline cJSON* target_level_description(cJSON* header,
                                       std::vector<size_t>* transpose) {
  std::vector<size_t> shape = header_shape(header);
  transpose->assign(shape.size(), 0);
  for (size_t d = 0; d < transpose->size(); ++d) {
    (*transpose)[d] = d;
  }

  const char* format = header_format(header);
  if (!strcmp(format, "custom")) {
    cJSON* custom = cJSON_GetObjectItemCaseSensitive(header, "custom");
    cJSON* level = cJSON_GetObjectItemCaseSensitive(custom, "level");
    if (!cJSON_IsObject(level)) {
      die("custom target requires custom.level");
    }
    cJSON* json = cJSON_GetObjectItemCaseSensitive(custom, "transpose");
    if (json != NULL) {
      for (size_t d = 0; d < transpose->size(); ++d) {
        (*transpose)[d] = (size_t) cJSON_GetNumberValue(
            cJSON_GetArrayItem(json, (int) d));
      }
    }
    return cJSON_Duplicate(level, 1);
  }

  cJSON* level =
      predefined_level(format, transpose->empty() ? NULL : transpose->data(),
                       (int) transpose->size());
  if (level == NULL) {
    die("unknown or rank-incompatible target format");
  }
  return level;
}

inline taco::Format taco_format_from_header(cJSON* header) {
  std::vector<size_t> transpose;
  cJSON* level = target_level_description(header, &transpose);
  std::vector<taco::ModeFormatPack> modes;
  append_taco_modes(level, &modes);
  cJSON_Delete(level);
  if (modes.empty()) {
    return taco::Format();
  }
  std::vector<int> mode_ordering(transpose.size());
  for (size_t d = 0; d < transpose.size(); ++d) {
    mode_ordering[d] = (int) transpose[d];
  }
  return taco::Format(modes, mode_ordering);
}

inline size_t dense_offset(const std::vector<size_t>& shape,
                           const size_t* coord) {
  size_t offset = 0;
  for (size_t d = 0; d < shape.size(); ++d) {
    offset = offset * shape[d] + coord[d];
  }
  return offset;
}

inline void write_taco_as_binsparse(const char* output,
                                    taco::TensorBase tensor,
                                    cJSON* target_header,
                                    const bsp_array_t* fill_value) {
  std::vector<size_t> target_transpose;
  cJSON* description = target_level_description(target_header, &target_transpose);
  const char* target_format = header_format(target_header);
  std::vector<int> dims = tensor.getDimensions();

  bsp_tensor_t converted = bsp_taco::makeBspTensor(tensor);
  entries_t entries = tensor_to_coo(&converted);
  if (!target_transpose.empty()) {
    apply_transpose(&entries, target_transpose.data(), target_transpose.data());
  }
  bsp_array_t converted_values = bsp_get_tensor_values(converted);
  bsp_array_t values = values_in_entry_order(converted_values, &entries, false);

  bsp_tensor_t result = bsp_construct_default_tensor_t();
  result.rank = (int) dims.size();
  result.nnz = entries.size;
  result.is_iso = false;
  result.structure = converted.structure;
  result.dims = result.rank ? (size_t*) malloc(dims.size() * sizeof(size_t))
                            : NULL;
  result.transpose =
      result.rank ? (size_t*) malloc(dims.size() * sizeof(size_t)) : NULL;
  std::vector<size_t> stored_dims(dims.size());
  for (int d = 0; d < result.rank; ++d) {
    result.dims[d] = (size_t) dims[(size_t) d];
    result.transpose[d] = target_transpose[(size_t) d];
    stored_dims[(size_t) d] = result.dims[result.transpose[d]];
  }

  reformat_context_t context = {
      cJSON_GetObjectItemCaseSensitive(target_header, "data_types")};
  size_t root_ptr[2] = {0, entries.size};
  result.level =
      build_level(&context, description, &entries,
                  stored_dims.empty() ? NULL : stored_dims.data(), 0, root_ptr,
                  1, true, values);

  if (!strcmp(target_format, "custom")) {
    if (bsp_write_tensor(output, result, NULL, "{}", 0) != BSP_SUCCESS) {
      die("failed to write output tensor");
    }
  } else if (write_predefined(output, result, target_format, 0) !=
             BSP_SUCCESS) {
    die("failed to write predefined output");
  }
  complete_output(output, target_header, fill_value);

  bsp_destroy_tensor_t(result);
  free_entries(&entries);
  bsp_destroy_tensor_t(converted);
  cJSON_Delete(description);
}
