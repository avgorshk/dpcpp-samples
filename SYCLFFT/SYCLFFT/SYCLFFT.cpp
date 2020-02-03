#include <chrono>
#include <iostream>
#include <vector>

#include <assert.h>
#include <string.h>

#include <CL/sycl.hpp>
#include <mkl_dfti_sycl.hpp>

#define IS_POW_2(X) (((X - 1) & X) == 0)

void Usage() {
  std::cout << "SYCLFFT.exe <size> <cpu|gpu>" << std::endl;
}

std::vector<float> GenerateSignal(int size) {
  std::vector<float> signal(size);
  for (auto& value : signal) {
    value = 0.001f + static_cast<float>(rand()) / RAND_MAX;
  }
  return signal;
}

float Verify(const std::vector<float>& computed,
             const std::vector<float>& expected) {
  assert(computed.size() == expected.size());

  float eps = 0.0f;
  for (size_t i = 0; i < computed.size(); ++i) {
    eps += fabsf((computed[i] - expected[i]) / expected[i]);
  }

  return eps / computed.size();
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    Usage();
    return 0;
  }

  int size = atoi(argv[1]);
  assert(size > 0);

  if (!IS_POW_2(size)) {
    std::cout << "Size should be power of two" << std::endl;
    return 0;
  }

  bool cpu = true;
  if (strcmp(argv[2], "gpu") == 0) {
    cpu = false;
  }

  std::unique_ptr<cl::sycl::device_selector> selector(nullptr);
  if (cpu) {
    selector.reset(new cl::sycl::cpu_selector);
  } else {
    selector.reset(new cl::sycl::gpu_selector);
  }

  std::vector<float> input = GenerateSignal(size);
  std::vector<float> output(input);

  cl::sycl::queue queue(*selector.get(), cl::sycl::async_handler{});
  std::cout << "Target device: " <<
    queue.get_info<cl::sycl::info::queue::device>().get_info<
    cl::sycl::info::device::name>() << std::endl;

  auto start = std::chrono::steady_clock::now();

  try {
    cl::sycl::buffer<float, 1> input_buf(input.data(), size + 2);

    mkl::dft::Descriptor<mkl::dft::Precision::SINGLE,
      mkl::dft::Domain::REAL> desc;

    std::vector<MKL_LONG> dim = { size };
    mkl::dft::ErrCode status = desc.init(dim);
    assert(status == mkl::dft::ErrCode::NO_ERROR);

    status = desc.setValue(mkl::dft::ConfigParam::BACKWARD_SCALE,
                           1.0f / size);
    assert(status == mkl::dft::ErrCode::NO_ERROR);

    status = desc.commit(queue);
    assert(status == mkl::dft::ErrCode::NO_ERROR);

    status = desc.computeForward(input_buf);
    assert(status == mkl::dft::ErrCode::NO_ERROR);
    status = desc.computeBackward(input_buf);
    assert(status == mkl::dft::ErrCode::NO_ERROR);

    queue.wait_and_throw();
  } catch (std::exception e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  auto end = std::chrono::steady_clock::now();
  float time =
    std::chrono::duration_cast<std::chrono::milliseconds>(
      end - start).count();
  std::cout << "Execution time: " << time << " ms" << std::endl;
  std::cout << "Difference: " << Verify(input, output);

  return 0;
}