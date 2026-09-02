/* SPDX-FileCopyrightText: 2026 Binsparse Developers
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "taco_reformat.hpp"

#include <binsparse/hdf5_wrapper.h>
#include <binsparse/read_tensor.h>
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

inline void die(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(1);
}

inline bool type_name_has_prefix(const char* name, const char* prefix) {
  return name != NULL && strncmp(name, prefix, strlen(prefix)) == 0;
}

inline bsp_type_t type_from_name(const char* name) {
  if (name == NULL) {
    return BSP_INVALID_TYPE;
  }
  if (type_name_has_prefix(name, "iso[")) {
    std::string inner(name + 4, strlen(name) - 5);
    return type_from_name(inner.c_str());
  }
  if (!strcmp(name, "uint8")) return BSP_UINT8;
  if (!strcmp(name, "uint16")) return BSP_UINT16;
  if (!strcmp(name, "uint32")) return BSP_UINT32;
  if (!strcmp(name, "uint64")) return BSP_UINT64;
  if (!strcmp(name, "int8")) return BSP_INT8;
  if (!strcmp(name, "int16")) return BSP_INT16;
  if (!strcmp(name, "int32")) return BSP_INT32;
  if (!strcmp(name, "int64")) return BSP_INT64;
  if (!strcmp(name, "float32")) return BSP_FLOAT32;
  if (!strcmp(name, "float64")) return BSP_FLOAT64;
  if (!strcmp(name, "bint8")) return BSP_BINT8;
  if (!strcmp(name, "complex64") || !strcmp(name, "complex[float32]")) {
    return BSP_COMPLEX_FLOAT32;
  }
  if (!strcmp(name, "complex128") || !strcmp(name, "complex[float64]")) {
    return BSP_COMPLEX_FLOAT64;
  }
  return BSP_INVALID_TYPE;
}

inline cJSON* input_header(const char* path) {
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    die("cannot open Binsparse file");
  }
  char* text = NULL;
  if (bsp_read_attribute(&text, file, "binsparse") != BSP_SUCCESS ||
      text == NULL) {
    H5Fclose(file);
    die("cannot read Binsparse metadata");
  }
  H5Fclose(file);

  cJSON* outer = cJSON_Parse(text);
  free(text);
  if (!cJSON_IsObject(outer)) {
    cJSON_Delete(outer);
    die("invalid Binsparse metadata");
  }
  cJSON* header = cJSON_DetachItemFromObjectCaseSensitive(outer, "binsparse");
  cJSON_Delete(outer);
  if (!cJSON_IsObject(header)) {
    cJSON_Delete(header);
    die("metadata has no binsparse header");
  }
  return header;
}

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

inline bool header_has_fill(cJSON* header) {
  return cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(header, "fill"));
}

inline bool header_values_are_iso(cJSON* header) {
  cJSON* data_types = cJSON_GetObjectItemCaseSensitive(header, "data_types");
  const char* value_name = cJSON_GetStringValue(
      cJSON_GetObjectItemCaseSensitive(data_types, "values"));
  return type_name_has_prefix(value_name, "iso[");
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

inline bsp_tensor_t custom_tensor_from_binsparse(const char* path,
                                                cJSON* header) {
  if (strcmp(header_format(header), "custom")) {
    die("predefined Binsparse aliases are not covered by the TACO harness");
  }

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
  bsp_tensor_t tensor = custom_tensor_from_binsparse(path, header);
  bsp_array_t fill = read_or_zero_fill_value(path, header);
  taco::TensorBase result = bsp_taco::makeTacoTensor(tensor, &fill);
  bsp_destroy_array_t(&fill);
  bsp_destroy_tensor_t(tensor);
  return result;
}

inline size_t dense_offset(const std::vector<size_t>& shape,
                           const size_t* coord) {
  size_t offset = 0;
  for (size_t d = 0; d < shape.size(); ++d) {
    offset = offset * shape[d] + coord[d];
  }
  return offset;
}

inline size_t array_value_as_size(const bsp_array_t& array, size_t index) {
  switch (array.type) {
  case BSP_UINT8: return static_cast<uint8_t*>(array.data)[index];
  case BSP_UINT16: return static_cast<uint16_t*>(array.data)[index];
  case BSP_UINT32: return static_cast<uint32_t*>(array.data)[index];
  case BSP_UINT64: return static_cast<uint64_t*>(array.data)[index];
  case BSP_INT8:
  case BSP_BINT8: return static_cast<int8_t*>(array.data)[index];
  case BSP_INT16: return static_cast<int16_t*>(array.data)[index];
  case BSP_INT32: return static_cast<int32_t*>(array.data)[index];
  case BSP_INT64: return static_cast<int64_t*>(array.data)[index];
  default: die("unsupported index type");
  }
  return 0;
}

inline size_t stored_dimension(const bsp_tensor_t& tensor, size_t depth) {
  size_t logical = tensor.transpose != NULL ? tensor.transpose[depth] : depth;
  if (logical >= (size_t) tensor.rank) {
    die("invalid tensor transpose");
  }
  return tensor.dims[logical];
}

inline size_t logical_dimension(const bsp_tensor_t& tensor, size_t depth) {
  size_t logical = tensor.transpose != NULL ? tensor.transpose[depth] : depth;
  if (logical >= (size_t) tensor.rank) {
    die("invalid tensor transpose");
  }
  return logical;
}

inline void collect_stored_coordinates_from_level(
    const bsp_tensor_t& tensor,
    const bsp_level_t* level,
    size_t depth,
    size_t parent,
    std::vector<size_t>* coord,
    std::vector<std::vector<size_t> >* coordinates);

inline void collect_dense_coordinates(const bsp_tensor_t& tensor,
                                      const bsp_dense_t* dense,
                                      size_t depth,
                                      size_t axis,
                                      size_t block,
                                      size_t local,
                                      size_t parent,
                                      std::vector<size_t>* coord,
                                      std::vector<std::vector<size_t> >*
                                          coordinates) {
  if (axis == (size_t) dense->rank) {
    collect_stored_coordinates_from_level(tensor, dense->child,
                                          depth + (size_t) dense->rank,
                                          parent * block + local, coord,
                                          coordinates);
    return;
  }

  size_t dimension = stored_dimension(tensor, depth + axis);
  for (size_t value = 0; value < dimension; ++value) {
    (*coord)[logical_dimension(tensor, depth + axis)] = value;
    collect_dense_coordinates(tensor, dense, depth, axis + 1, block,
                              local * dimension + value, parent, coord,
                              coordinates);
  }
}

inline void collect_stored_coordinates_from_level(
    const bsp_tensor_t& tensor,
    const bsp_level_t* level,
    size_t depth,
    size_t parent,
    std::vector<size_t>* coord,
    std::vector<std::vector<size_t> >* coordinates) {
  switch (level->kind) {
  case BSP_TENSOR_ELEMENT: {
    const bsp_element_t* element = (const bsp_element_t*) level->data;
    if (coord->empty()) {
      for (size_t i = 0; i < element->values.size; ++i) {
        coordinates->push_back(*coord);
      }
    } else {
      coordinates->push_back(*coord);
    }
    return;
  }
  case BSP_TENSOR_DENSE: {
    const bsp_dense_t* dense = (const bsp_dense_t*) level->data;
    size_t block = 1;
    for (int axis = 0; axis < dense->rank; ++axis) {
      block *= stored_dimension(tensor, depth + (size_t) axis);
    }
    collect_dense_coordinates(tensor, dense, depth, 0, block, 0, parent, coord,
                              coordinates);
    return;
  }
  case BSP_TENSOR_SPARSE: {
    const bsp_sparse_t* sparse = (const bsp_sparse_t*) level->data;
    size_t begin = 0;
    size_t end = sparse->indices[0].size;
    if (sparse->pointers_to != NULL) {
      begin = array_value_as_size(*sparse->pointers_to, parent);
      end = array_value_as_size(*sparse->pointers_to, parent + 1);
    }
    for (size_t pos = begin; pos < end; ++pos) {
      for (int axis = 0; axis < sparse->rank; ++axis) {
        (*coord)[logical_dimension(tensor, depth + (size_t) axis)] =
            array_value_as_size(sparse->indices[axis], pos);
      }
      collect_stored_coordinates_from_level(
          tensor, sparse->child, depth + (size_t) sparse->rank, pos, coord,
          coordinates);
    }
    return;
  }
  default:
    die("unknown Binsparse tensor level");
  }
}

inline std::vector<std::vector<size_t> >
stored_coordinates(taco::TensorBase tensor) {
  bsp_tensor_t converted = bsp_taco::makeBspTensor(tensor);
  std::vector<std::vector<size_t> > coordinates;
  std::vector<size_t> coord((size_t) converted.rank, 0);
  collect_stored_coordinates_from_level(converted, converted.level, 0, 0,
                                        &coord, &coordinates);
  bsp_destroy_tensor_t(converted);
  return coordinates;
}

inline void copy_binsparse_file(const char* source, const char* output) {
  FILE* input = fopen(source, "rb");
  if (input == NULL) {
    die("cannot open input Binsparse file");
  }
  FILE* result = fopen(output, "wb");
  if (result == NULL) {
    fclose(input);
    die("cannot open output Binsparse file");
  }

  char buffer[8192];
  while (true) {
    size_t count = fread(buffer, 1, sizeof(buffer), input);
    if (count != 0 && fwrite(buffer, 1, count, result) != count) {
      fclose(result);
      fclose(input);
      die("cannot write output Binsparse file");
    }
    if (count < sizeof(buffer)) {
      if (ferror(input)) {
        fclose(result);
        fclose(input);
        die("cannot read input Binsparse file");
      }
      break;
    }
  }

  fclose(result);
  fclose(input);
}
