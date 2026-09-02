#include <binsparse/tensor.h>
#include <binsparse/write_tensor.h>
#include <taco.h>

#pragma once

namespace bsp_taco {
bsp_tensor_t makeBspTensor(taco::TensorBase tacoTensor);

void writeBinSparse(taco::TensorBase taco, std::string filename);
}
