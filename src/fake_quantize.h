#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <ATen/ATen.h>
#include <torch/torch.h>

using namespace at;

Tensor fake_quantize(Tensor a, int bit_width=8);

Tensor fake_quantize_cuda(Tensor a, int bit_width=8);

__global__ void fake_quantize_kernel_cuda(float* __restrict__ a,
                                            float* o, int size,
                                            float* max_entry,
                                            int bit_width=8);
