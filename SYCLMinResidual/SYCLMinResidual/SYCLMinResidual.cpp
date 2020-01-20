#include <iostream>
#include <chrono>
#include <vector>

#include <assert.h>
#include <math.h>
#include <string.h>

#include <CL/sycl.hpp>

#define AVALUE 2.4f
#define BVALUE 1024.2f

std::vector<float> ComputeNative(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 bool cpu) {
  assert(a.size() == b.size() * b.size());
  size_t size = b.size();

  std::vector<float> x1(size, 0.0f);
  std::vector<float> x2(size, 0.0f);

  cl::sycl::buffer<float, 1> a_buf(a.data(), a.size());
  cl::sycl::buffer<float, 1> b_buf(b.data(), b.size());
  cl::sycl::buffer<float, 1> x1_buf(x1.data(), x1.size());
  cl::sycl::buffer<float, 1> x2_buf(x2.data(), x2.size());

  std::unique_ptr<cl::sycl::device_selector> selector(nullptr);
  if (cpu) {
    selector.reset(new cl::sycl::cpu_selector);
  } else {
    selector.reset(new cl::sycl::gpu_selector);
  }

  try {
    cl::sycl::queue queue(*selector.get(), cl::sycl::async_handler{});
    std::cout << "Target device: " <<
      queue.get_info<cl::sycl::info::queue::device>().get_info<
      cl::sycl::info::device::name>() << std::endl;

    queue.submit([&](cl::sycl::handler& cgh) {
      auto a_acc = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto b_acc = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto x1_acc = x1_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto x2_acc = x2_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

      //cgh.parallel_for<class VectorAdd>(num_items, [=](nd_item<1> wiID) {
      //});
    });

  } catch (std::exception e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  return std::vector<float>(b.size(), 0.0f);
}

std::vector<float> GenerateMatrix(int size) {
  std::vector<float> matrix(size * size);
  for (auto& value : matrix) {
    value = AVALUE;
  }
  for (size_t i = 0; i < size; ++i) {
    matrix[i * size + i] = size * AVALUE + AVALUE;
  }
  return matrix;
}

std::vector<float> GenerateVector(int size) {
  std::vector<float> vector(size);
  for (auto& value : vector) {
    value = BVALUE;
  }
  return vector;
}

float ComputeError(const std::vector<float>& a, const std::vector<float>& b,
                   const std::vector<float>& x) {
  assert(b.size() == x.size());
  assert(b.size() * b.size() == a.size());

  size_t size = x.size();
  std::vector<float> eps(size);
  
  for (size_t i = 0; i < size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < size; ++j) {
      sum += a[i * size + j] * x[j];
    }
    if (sum > 0.0e-5f) {
      eps[i] = fabsf((sum - b[i]) / sum);
    } else if (b[i] > 0.0e-5f) {
      eps[i] = fabsf((sum - b[i]) / b[i]);
    } else {
      eps[i] = 0.0f;
    }
  }

  float avg = 0.0f;
  for (auto value : eps) {
    avg += value;
  }
  return avg / eps.size();
}

void Usage() {
  std::cout << "SYCLMinResidual.exe <matrix_size> <iteration_count>" <<
    " <native|mkl> <cpu|gpu>" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    Usage();
    return 0;
  }

  int size = atoi(argv[1]);
  assert(size > 0);
  int count = atoi(argv[2]);
  assert(count > 0);

  bool mkl = false;
  if (strcmp(argv[3], "mkl") == 0) {
    mkl = true;
  }

  bool cpu = false;
  if (strcmp(argv[4], "cpu") == 0) {
    cpu = true;
  }

  std::vector<float> a = GenerateMatrix(size);
  std::vector<float> b = GenerateVector(size);
  std::vector<float> x;

  auto start = std::chrono::steady_clock::now();
  if (mkl) {

  } else {
    x = ComputeNative(a, b, cpu);
  }
  auto end = std::chrono::steady_clock::now();
  float time =
    std::chrono::duration_cast<std::chrono::milliseconds>(
      end - start).count();
  std::cout << "Execution time: " << time << " ms" << std::endl;

  float eps = ComputeError(a, b, x);
  std::cout << "Error: " << eps << std::endl;

  return 0;
}