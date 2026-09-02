#include <bsp_taco/bsp_to_taco.hpp>

#include <complex>
#include <cstring>

static std::vector<int> getDimensions(bsp_tensor_t& tensor) {
  std::vector<int> dims(tensor.rank);
  for (int i = 0; i < tensor.rank; i++) {
    dims[i] = tensor.dims[i];
  }
  return dims;
}

static inline taco::Datatype getTacoDataType(bsp_array_t& array) {
  bsp_type_t type = array.type;
  switch (type) {
  case BSP_UINT8:
    return taco::UInt8;
  case BSP_UINT16:
    return taco::UInt16;
  case BSP_UINT32:
    return taco::UInt32;
  case BSP_UINT64:
    return taco::UInt64;
  case BSP_INT8:
    return taco::Int8;
  case BSP_INT16:
    return taco::Int16;
  case BSP_INT32:
    return taco::Int32;
  case BSP_INT64:
    return taco::Int64;
  case BSP_FLOAT32:
    return taco::Float32;
  case BSP_FLOAT64:
    return taco::Float64;
  case BSP_BINT8:
    return taco::Int8;
  case BSP_COMPLEX_FLOAT32:
    return taco::Complex64;
  case BSP_COMPLEX_FLOAT64:
    return taco::Complex128;
  default:
    taco_uerror << "Unsupported type supplied to taco converter";
    return taco::Float64;
  }
}

static taco::Array bspToTacoArray(bsp_array_t& array) {
  taco::Datatype dataType = getTacoDataType(array);
  bsp_type_t type = array.type;
  taco::Array res = taco::makeArray(dataType, array.size);
  // eventually, get rid of memcpy?
  memcpy(res.getData(), array.data, array.size * dataType.getNumBytes());
  return res;
}

static taco::Format createTacoFormat(bsp_tensor_t& tensor) {
  bsp_level_t* level = tensor.level;
  std::vector<taco::ModeFormatPack> modeTypes;
  while (level->kind != BSP_TENSOR_ELEMENT) {
    switch (level->kind) {
    case BSP_TENSOR_DENSE: {
      modeTypes.push_back(taco::Dense());
      level = ((bsp_dense_t*) level->data)->child;
      break;
    }
    case BSP_TENSOR_SPARSE: {
      bsp_sparse_t* data = (bsp_sparse_t*) level->data;
      modeTypes.push_back(taco::Compressed(
          {taco::ModeFormat::ORDERED, taco::ModeFormat::NOT_UNIQUE}));
      for (int i = 1; i < data->rank; i++) {
        modeTypes.push_back(taco::Singleton(
            {taco::ModeFormat::ORDERED, taco::ModeFormat::NOT_UNIQUE}));
      }
      level = ((bsp_sparse_t*) level->data)->child;
      break;
    }
    default:;
    }
  }
  if (tensor.transpose != NULL) {
    std::vector<int> modeOrdering(tensor.rank);
    for (int i = 0; i < tensor.rank; i++) {
      modeOrdering[i] = tensor.transpose[i];
    }
    return taco::Format(modeTypes, modeOrdering);
  }
  return taco::Format(modeTypes);
}

static taco::Index createTacoIndex(bsp_tensor_t& tensor, taco::Format& format) {
  bsp_level_t* level = tensor.level;
  size_t* dims = tensor.dims;
  std::vector<taco::ModeIndex> modeIndices;
  while (level->kind != BSP_TENSOR_ELEMENT) {
    switch (level->kind) {
    case BSP_TENSOR_DENSE: {
      modeIndices.push_back(taco::ModeIndex({taco::makeArray({(int) *dims})}));
      level = ((bsp_dense_t*) level->data)->child;
      break;
    }
    case BSP_TENSOR_SPARSE: {
      bsp_sparse_t* data = (bsp_sparse_t*) level->data;
      if (data->pointers_to != NULL && data->pointers_to->type != BSP_INT32) {
        taco_uerror << "pointers_to just be an int32 type to interface "
                    << "properly with taco!";
      }
      modeIndices.push_back(taco::ModeIndex({
          data->pointers_to != NULL
              ? taco::Array(getTacoDataType(*data->pointers_to),
                            data->pointers_to->data, data->pointers_to->size,
                            taco::Array::Free)
              : taco::makeArray({
                    (int) 0,
                    (int) data->indices[0].size,
                }),
          taco::Array(getTacoDataType(data->indices[0]), data->indices[0].data,
                      data->indices[0].size, taco::Array::Free),
      }));
      for (int i = 1; i < data->rank; i++) {
        if (data->indices[i].type != BSP_INT32)
          taco_terror << "indices just be an int32 type to interface properly "
                      << "with taco!";
        modeIndices.push_back(taco::ModeIndex({
            taco::makeArray(getTacoDataType(data->indices[i]), 0),
            taco::Array(getTacoDataType(data->indices[i]),
                        data->indices[i].data, data->indices[i].size,
                        taco::Array::Free),
        }));
      }
      level = ((bsp_sparse_t*) level->data)->child;
      break;
    }
    default:;
    }
    dims++;
  }
  return taco::Index(format, modeIndices);
}

template <typename T>
static taco::Literal scalarLiteral(const bsp_array_t& array) {
  T value;
  memcpy(&value, array.data, sizeof(value));
  return taco::Literal(value);
}

static taco::Literal getFillLiteral(const bsp_array_t& fill) {
  switch (fill.type) {
  case BSP_UINT8: return scalarLiteral<uint8_t>(fill);
  case BSP_UINT16: return scalarLiteral<uint16_t>(fill);
  case BSP_UINT32: return scalarLiteral<uint32_t>(fill);
  case BSP_UINT64: return scalarLiteral<uint64_t>(fill);
  case BSP_INT8:
  case BSP_BINT8: return scalarLiteral<int8_t>(fill);
  case BSP_INT16: return scalarLiteral<int16_t>(fill);
  case BSP_INT32: return scalarLiteral<int32_t>(fill);
  case BSP_INT64: return scalarLiteral<int64_t>(fill);
  case BSP_FLOAT32: return scalarLiteral<float>(fill);
  case BSP_FLOAT64: return scalarLiteral<double>(fill);
  case BSP_COMPLEX_FLOAT32:
    return scalarLiteral<std::complex<float>>(fill);
  case BSP_COMPLEX_FLOAT64:
    return scalarLiteral<std::complex<double>>(fill);
  default:
    taco_uerror << "Unsupported fill type supplied to taco converter";
    return taco::Literal();
  }
}

/*
Creates a taco object from a bsp tensor.
Note that this function **consumes** the bsp tensor object!
*/
taco::TensorBase bsp_taco::makeTacoTensor(bsp_tensor_t& tensor,
                                          const bsp_array_t* fill) {
  if (tensor.is_iso)
    taco_uerror << "TACO does not support iso-valued tensor storage";
  bsp_level_t* level = tensor.level;

  bsp_array_t values = bsp_get_tensor_values(tensor);
  taco::Format tacoFormat = createTacoFormat(tensor);
  taco::Index tacoIndex = createTacoIndex(tensor, tacoFormat);
  taco::Literal tacoFill =
      fill == nullptr ? taco::Literal() : getFillLiteral(*fill);
  taco::TensorBase tacoTensor(getTacoDataType(values), getDimensions(tensor),
                              tacoFormat, tacoFill);
  // tacoTensor.setNeedsPack(false);
  auto storage = tacoTensor.getStorage();
  storage.setIndex(tacoIndex);
  storage.setValues(bspToTacoArray(values));
  tacoTensor.setStorage(storage);
  return tacoTensor;
}

taco::TensorBase bsp_taco::readBinSparse(std::string filename) {
  bsp_tensor_t tensor = bsp_read_tensor(filename.data(), NULL);
  return bsp_taco::makeTacoTensor(tensor);
}
