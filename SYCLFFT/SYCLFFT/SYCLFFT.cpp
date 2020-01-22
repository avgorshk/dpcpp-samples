#include <iostream>
#include <vector>

#include <assert.h>
#include <string.h>

#include <CL/sycl.hpp>
#include <mkl_dfti_sycl.hpp>

#define VALUE 0.128f

void Usage() {
  std::cout << "SYCLFFT.exe <size> <cpu|gpu>" << std::endl;
}

std::vector<float> GenerateSignal(int size) {
  std::vector<float> signal(size);
  for (auto& value : signal) {
    value = VALUE;
  }
  return signal;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    Usage();
    return 0;
  }

  int size = atoi(argv[1]);
  assert(size > 0);

  bool cpu = true;
  if (strcmp(argv[2], "gpu") == 0) {
    cpu = false;
  }

  std::vector<float> signal = GenerateSignal(size);

  std::unique_ptr<cl::sycl::device_selector> selector(nullptr);
  if (cpu) {
    selector.reset(new cl::sycl::cpu_selector);
  } else {
    selector.reset(new cl::sycl::gpu_selector);
  }

  cl::sycl::queue queue(*selector.get(), cl::sycl::async_handler{});
  std::cout << "Target device: " <<
    queue.get_info<cl::sycl::info::queue::device>().get_info<
    cl::sycl::info::device::name>() << std::endl;

  mkl::dft::Descriptor<mkl::dft::Precision::SINGLE,
                       mkl::dft::Domain::REAL> desc;

  mkl::dft::ErrCode status = desc.init(size);
  assert(status == mkl::dft::ErrCode::NO_ERROR);

  return 0;
}