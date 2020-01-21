#include <iostream>
#include <chrono>
#include <vector>

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <CL/sycl.hpp>
#include <mkl_blas_sycl.hpp>

#define EPS 1.0e-5f

using HostReadAccessor = cl::sycl::accessor<float, 1,
  cl::sycl::access::mode::read,
  cl::sycl::access::target::host_buffer>;

class ComputeR;
class ComputeAr;
class ComputeX;

std::vector<float> Transpose(const std::vector<float>& a, int size) {
  assert(size * size == a.size());
  std::vector<float> at(size * size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      at[i * size + j] = a[j * size + i];
    }
  }

  return at;
}

float Dot(const HostReadAccessor& a,
          const HostReadAccessor& b) {
  assert(a.get_count() == b.get_count());
  
  float dot = 0.0f;
  for (size_t i = 0; i < a.get_count(); ++i) {
    dot += a[i] * b[i];
  }

  return dot;
}

std::vector<float> ComputeNative(cl::sycl::queue& queue,
                                 const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int& count) {
  assert(a.size() == b.size() * b.size());
  size_t size = b.size();

  std::vector<float> at = Transpose(a, size);
  std::vector<float> x(size, 0.0f);
  std::vector<float> r(size, 0.0f);
  std::vector<float> ar(size, 0.0f);

  try {
    cl::sycl::buffer<float, 1> at_buf(at.data(), at.size());
    cl::sycl::buffer<float, 1> b_buf(b.data(), b.size());
    cl::sycl::buffer<float, 1> x_buf(x.data(), x.size());
    cl::sycl::buffer<float, 1> r_buf(r.data(), r.size());
    cl::sycl::buffer<float, 1> ar_buf(ar.data(), ar.size());

    for (int it = 0; it < count; ++it) {

      // r <- Ax - b
      queue.submit([&](cl::sycl::handler& cgh) {
        auto at_acc = at_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class ComputeR>(cl::sycl::range<1>(size),
                                         [=](cl::sycl::id<1> id) {
          float sum = 0.0f;
          for (int i = 0; i < size; ++i) {
            sum += at_acc[i * size + id] * x_acc[i];
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
        auto at_acc = at_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto r_acc = r_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto ar_acc = ar_buf.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class ComputeAr>(cl::sycl::range<1>(size),
                                          [=](cl::sycl::id<1> id) {
          float sum = 0.0f;
          for (int i = 0; i < size; ++i) {
            sum += at_acc[i * size + id] * r_acc[i];
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

std::vector<float> ComputeMKL(cl::sycl::queue& queue,
                              const std::vector<float>& a,
                              const std::vector<float>& b,
                              int& count) {
  assert(a.size() == b.size() * b.size());
  size_t size = b.size();
  
  std::vector<float> at = Transpose(a, size);
  std::vector<float> x(size, 0.0f);
  std::vector<float> r(size, 0.0f);
  std::vector<float> ar(size, 0.0f);

  try {
    cl::sycl::buffer<float, 1> a_buf(at.data(), at.size());
    cl::sycl::buffer<float, 1> b_buf(b.data(), b.size());
    cl::sycl::buffer<float, 1> x_buf(x.data(), x.size());
    cl::sycl::buffer<float, 1> r_buf(r.data(), r.size());
    cl::sycl::buffer<float, 1> ar_buf(ar.data(), ar.size());

    float ar_r = 0.0f, ar_ar = 0.0f, norm = 0.0f, tau = 0.0f;
    cl::sycl::buffer<float, 1> ar_r_buf(&ar_r, 1);
    cl::sycl::buffer<float, 1> ar_ar_buf(&ar_ar, 1);
    cl::sycl::buffer<float, 1> norm_buf(&norm, 1);

    for (int it = 0; it < count; ++it) {
      // r <- Ax - b
      mkl::blas::copy(queue, size, b_buf, 1, r_buf, 1);
      mkl::blas::gemv(queue, mkl::transpose::N, size, size, 1.0f, a_buf, size, x_buf, 1, -1.0f, r_buf, 1);
      mkl::blas::dot(queue, size, r_buf, 1, r_buf, 1, norm_buf);
      auto norm_acc = norm_buf.get_access<cl::sycl::access::mode::read>();
      if (sqrtf(norm_acc[0]) < EPS) {
        count = it;
        break;
      }

      // tau = (Ar, r) / (Ar, Ar)
      mkl::blas::gemv(queue, mkl::transpose::N, size, size, 1.0f, a_buf, size, r_buf, 1, 0.0f, ar_buf, 1);
      mkl::blas::dot(queue, size, ar_buf, 1, r_buf, 1, ar_r_buf);
      mkl::blas::dot(queue, size, ar_buf, 1, ar_buf, 1, ar_ar_buf);
      auto ar_r_acc = ar_r_buf.get_access<cl::sycl::access::mode::read>();
      auto ar_ar_acc = ar_ar_buf.get_access<cl::sycl::access::mode::read>();
      tau = ar_r_acc[0] / ar_ar_acc[0];

      // x <- x - tau * r
      mkl::blas::axpy(queue, size, -tau, r_buf, 1, x_buf, 1);
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

  std::cout << "Warming-up...";
  int temp_count = 1;
  if (mkl) {
    ComputeMKL(queue, a, b, temp_count);
  } else {
    ComputeNative(queue, a, b, temp_count);
  }
  std::cout << "OK" << std::endl;

  std::cout << "Target mode: ";
  auto start = std::chrono::steady_clock::now();
  if (mkl) {
    std::cout << "MKL" << std::endl;
    x = ComputeMKL(queue, a, b, count);
  } else {
    std::cout << "Native" << std::endl;
    x = ComputeNative(queue, a, b, count);
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