#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fi_uint8_cuda(
    torch::Tensor input,
    float f01,
    float f10
  );

std::vector<torch::Tensor> fi_uint8(
    torch::Tensor input,
    float f01,
    float f10
  ) {
  CHECK_INPUT(input);
  return fi_uint8_cuda(input, f01, f10);
}


std::vector<torch::Tensor> fi_uint16_cuda(
    torch::Tensor input,
    float f01,
    float f10
  );

std::vector<torch::Tensor> fi_uint16(
    torch::Tensor input,
    float f01,
    float f10
  ) {
  CHECK_INPUT(input);
  return fi_uint16_cuda(input, f01, f10);
}


std::vector<torch::Tensor> fi_uint32_cuda(
    torch::Tensor input,
    float f01,
    float f10
  );

std::vector<torch::Tensor> fi_uint32(
    torch::Tensor input,
    float f01,
    float f10
  ) {
  CHECK_INPUT(input);
  return fi_uint32_cuda(input, f01, f10);
}


std::vector<torch::Tensor> fi_uint64_cuda(
    torch::Tensor input,
    float f01,
    float f10
  );

std::vector<torch::Tensor> fi_uint64(
    torch::Tensor input,
    float f01,
    float f10
  ) {
  CHECK_INPUT(input);
  return fi_uint64_cuda(input, f01, f10);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bfi_8bit", &fi_uint8, "BFI_8BIT");
  m.def("bfi_16bit", &fi_uint16, "BFI_16BIT");
  m.def("bfi_32bit", &fi_uint32, "BFI_32BIT");
  m.def("bfi_64bit", &fi_uint64, "BFI_64BIT");
}
