#include <binsparse/read_tensor.h>
#include <binsparse/tensor.h>
#include <taco.h>

#pragma once

namespace bsp_taco {
taco::TensorBase makeTacoTensor(bsp_tensor_t& tensor,
                                const bsp_array_t* fill = nullptr);

taco::TensorBase readBinSparse(std::string filename);
}
