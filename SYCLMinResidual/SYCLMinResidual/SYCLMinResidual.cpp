#include <iostream>
#include <chrono>
#include <vector>

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <CL/sycl.hpp>

#define EPS 1.0e-5f

using HostReadAccessor = cl::sycl::accessor<float, 1,
  cl::sycl::access::mode::read,
  cl::sycl::access::target::host_buffer>;

class ComputeR;
class ComputeAr;
class ComputeX;

float Dot(const HostReadAccessor& a,
          const HostReadAccessor& b) {
  assert(a.get_count() == b.get_count());
  
  float dot = 0.0f;
  for (size_t i = 0; i < a.get_count(); ++i) {
    dot += a[i] * b[i];
  }

  return dot;
}

std::vector<float> ComputeNative(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int& count, bool cpu) {
  assert(a.size() == b.size() * b.size());
  size_t size = b.size();

  std::vector<float> x(size, 0.0f);
  std::vector<float> r(size, 0.0f);
  std::vector<float> ar(size, 0.0f);

  std::unique_ptr<cl::sycl::device_selector> selector(nullptr);
  if (cpu) {
    selector.reset(new cl::sycl::cpu_selector);
  } else {
    selector.reset(new cl::sycl::gpu_selector);
  }

  try {
    cl::sycl::buffer<float, 1> a_buf(a.data(), a.size());
    cl::sycl::buffer<float, 1> b_buf(b.data(), b.size());
    cl::sycl::buffer<float, 1> x_buf(x.data(), x.size());
    cl::sycl::buffer<float, 1> r_buf(r.data(), r.size());
    cl::sycl::buffer<float, 1> ar_buf(ar.data(), ar.size());

    cl::sycl::queue queue(*selector.get(), cl::sycl::async_handler{});
    std::cout << "Target device: " <<
      queue.get_info<cl::sycl::info::queue::device>().get_info<
        cl::sycl::info::device::name>() << std::endl;

    for (int it = 0; it < count; ++it) {

      // r <- Ax - b
      queue.submit([&](cl::sycl::handler& cgh) {
        auto a_acc = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class ComputeR>(cl::sycl::range<1>(size),
                                          [=](cl::sycl::id<1> id) {
          float sum = 0.0f;
          for (int i = 0; i < size; ++i) {
            sum += a_acc[id * size + i] * x_acc[i];
          }
          r_acc[id] = sum - b_acc[id];
        });
      });

      {
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::read>();
        float norm = sqrtf(Dot(r_acc, r_acc));
        if (norm < EPS) {
          count = it;
          break;
        }
      }

      // Ar
      queue.submit([&](cl::sycl::handler& cgh) {
        auto a_acc = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto ar_acc = ar_buf.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class ComputeAr>(cl::sycl::range<1>(size),
                                          [=](cl::sycl::id<1> id) {
          float sum = 0.0f;
          for (int i = 0; i < size; ++i) {
            sum += a_acc[id * size + i] * r_acc[i];
          }
          ar_acc[id] = sum;
        });
      });

      // tau <- (Ar, r) / (Ar, Ar)
      float tau = 0.0f;
      {
        auto ar_acc = ar_buf.get_access<cl::sycl::access::mode::read>();
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::read>();
        float ar_r = Dot(ar_acc, r_acc);
        float ar_ar = Dot(ar_acc, ar_acc);
        assert(ar_ar > 0.0f);
        tau = ar_r / ar_ar;
      }

      // x <- x - tau * r
      queue.submit([&](cl::sycl::handler& cgh) {
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class ComputeX>(cl::sycl::range<1>(size),
                                          [=](cl::sycl::id<1> id) {
          x_acc[id] -= tau * r_acc[id];
        });
      });

    }

    queue.wait_and_throw();
  } catch (std::exception e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  return x;
}

std::vector<float> GenerateMatrix(int size) {
  std::vector<float> matrix(size * size);
  for (auto& value : matrix) {
    value = static_cast<float>(rand()) / RAND_MAX;
  }
  for (size_t i = 0; i < size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < size; ++j) {
      sum += matrix[i * size + j];
    }
    matrix[i * size + i] = 2.0f * sum;
  }
  return matrix;
}

std::vector<float> GenerateVector(int size) {
  std::vector<float> vector(size);
  for (auto& value : vector) {
    value = static_cast<float>(rand()) / RAND_MAX;
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

  srand(777);
  std::vector<float> a = GenerateMatrix(size);
  std::vector<float> b = GenerateVector(size);
  std::vector<float> x;

  auto start = std::chrono::steady_clock::now();
  if (mkl) {

  } else {
    x = ComputeNative(a, b, count, cpu);
  }
  auto end = std::chrono::steady_clock::now();
  float time =
    std::chrono::duration_cast<std::chrono::milliseconds>(
      end - start).count();
  std::cout << "Execution time: " << time << " ms" << std::endl;

  float eps = ComputeError(a, b, x);
  std::cout << "Error " << eps << " in " << count <<
    " iterations" << std::endl;

  return 0;
}