#include <bsp_taco/taco_to_bsp.hpp>

static inline bsp_type_t getTacoDataType(taco::Datatype type) {
  if (type == taco::UInt8)
    return BSP_UINT8;
  else if (type == taco::UInt16)
    return BSP_UINT16;
  else if (type == taco::UInt32)
    return BSP_UINT32;
  else if (type == taco::UInt64)
    return BSP_UINT64;
  else if (type == taco::Int8)
    return BSP_INT8;
  else if (type == taco::Int16)
    return BSP_INT16;
  else if (type == taco::Int32)
    return BSP_INT32;
  else if (type == taco::Int64)
    return BSP_INT64;
  else if (type == taco::Float32)
    return BSP_FLOAT32;
  else if (type == taco::Float64)
    return BSP_FLOAT64;
  else if (type == taco::Int8)
    return BSP_BINT8;
  else if (type == taco::Complex64)
    return BSP_COMPLEX_FLOAT32;
  else if (type == taco::Complex128)
    return BSP_COMPLEX_FLOAT64;
  else {
    taco_uerror << "Unsupported type supplied to taco converter";
    return BSP_INVALID_TYPE;
  }
}

static bsp_array_t makeBspIndexArray(taco::Array arr) {
  taco_uassert(arr.getType() == taco::Int32);
  bsp_array_t res;
  if (bsp_construct_array_t(&res, arr.getSize(), BSP_INT32) != BSP_SUCCESS)
    taco_uerror << "Unable to allocate Binsparse index array";
  memcpy(res.data, arr.getData(), taco::Int32.getNumBytes() * arr.getSize());
  return res;
}

bsp_tensor_t bsp_taco::makeBspTensor(taco::TensorBase tacoTensor) {
  bsp_tensor_t res = bsp_construct_default_tensor_t();
  auto storage = tacoTensor.getStorage();
  auto index = storage.getIndex();

  // copy over the transposes.
  auto modeOrdering = storage.getFormat().getModeOrdering();
  res.transpose = (size_t*) malloc(sizeof(size_t) * modeOrdering.size());
  for (int i = 0; i < modeOrdering.size(); i++) {
    res.transpose[i] = (size_t) modeOrdering[i];
  }

  // copy over the dimensions.
  auto dims = storage.getDimensions();
  res.rank = dims.size();
  res.dims = (size_t*) malloc(sizeof(size_t) * dims.size());
  for (int i = 0; i < dims.size(); i++) {
    res.dims[i] = dims[i];
  }

  std::vector<taco::ModeFormat> formats = storage.getFormat().getModeFormats();

  int dimsPtr = 0;
  res.level = (bsp_level_t*) malloc(sizeof(bsp_level_t));
  bsp_level_t* curLevel = res.level;

  while (dimsPtr < dims.size()) {
    taco::ModeFormat format = formats[dimsPtr];

    if (format.getName() == taco::Sparse.getName()) {
      int boundary = dimsPtr + 1;
      while (boundary < dims.size() &&
             formats[boundary].getName() == taco::Singleton.getName()) {
        boundary++;
      }
      curLevel->kind = BSP_TENSOR_SPARSE;

      bsp_sparse_t* data = (bsp_sparse_t*) malloc(sizeof(bsp_sparse_t));
      curLevel->data = data;

      data->pointers_to = NULL;
      if (dimsPtr != 0) {
        data->pointers_to = (bsp_array_t*) malloc(sizeof(bsp_array_t));
        *data->pointers_to =
            makeBspIndexArray(index.getModeIndex(dimsPtr).getIndexArray(0));
      }

      data->rank = boundary - dimsPtr;
      data->indices = (bsp_array_t*) malloc(sizeof(bsp_array_t) * data->rank);
      for (int indicesIdx = 0; indicesIdx < data->rank; indicesIdx++) {
        data->indices[indicesIdx] = makeBspIndexArray(
            index.getModeIndex(dimsPtr + indicesIdx).getIndexArray(1));
      }

      data->child = (bsp_level_t*) malloc(sizeof(bsp_level_t));
      curLevel = data->child;

      dimsPtr = boundary;
    } else if (format.getName() == taco::Dense.getName()) {
      curLevel->kind = BSP_TENSOR_DENSE;
      bsp_dense_t* data = (bsp_dense_t*) malloc(sizeof(bsp_dense_t));
      curLevel->data = data;

      data->rank = 1;
      data->child = (bsp_level_t*) malloc(sizeof(bsp_level_t));
      curLevel = data->child;

      dimsPtr++;
    } else {
      taco_uerror << "This should be impossible; neither dense nor sparse";
    }
  }

  // code to deal with copying over the actual data.
  {
    curLevel->kind = BSP_TENSOR_ELEMENT;
    taco::Array values = storage.getValues();
    res.nnz = values.getSize();
    bsp_element_t* data = (bsp_element_t*) malloc(sizeof(bsp_element_t));

    bsp_array_t valuesArray;
    if (bsp_construct_array_t(&valuesArray, res.nnz,
                              getTacoDataType(values.getType())) !=
        BSP_SUCCESS)
      taco_uerror << "Unable to allocate Binsparse values array";
    memcpy(valuesArray.data, values.getData(),
           values.getType().getNumBytes() * values.getSize());

    data->values = valuesArray;
    curLevel->data = data;
  }
  return res;
}

void bsp_taco::writeBinSparse(taco::TensorBase taco, std::string filename) {
  bsp_tensor_t tensor = bsp_taco::makeBspTensor(taco);
  if (bsp_write_tensor(filename.data(), tensor, NULL, NULL, 9) != BSP_SUCCESS)
    taco_uerror << "Unable to write Binsparse tensor";
  bsp_destroy_tensor_t(tensor);
}
