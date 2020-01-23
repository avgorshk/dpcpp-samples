#include <iostream>
#include <chrono>

#include <assert.h>
#include <math.h>
#include <string.h>

#include <CL/sycl.hpp>
#include <mkl_rng_sycl.hpp>

#define A_VALUE 0.0f
#define B_VALUE 1.0f

class ComputeIntegral;

void Usage() {
  std::cout << "SYCLIntegral.exe <dot_count> <cpu|gpu>" << std::endl;
}

float IntegralFunction(float x, float y) {
  return cl::sycl::sin(x) * cl::sycl::cos(y);
}

float GetIntegralValue(float a, float b) {
  return sinf(a) * (cosf(b) - cosf(a)) + sinf(b) * (cosf(a) - cosf(b));
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    Usage();
    return 0;
  }

  int count = atoi(argv[1]);
  assert(count > 0);

  bool cpu = true;
  if (strcmp(argv[2], "gpu") == 0) {
    cpu = false;
  }

  std::vector<float> dots(3 * count);
  int reached = 0;

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

  mkl::rng::sobol engine(queue, 3);
  mkl::rng::uniform<float, mkl::rng::method::standard>
    distribution(A_VALUE, B_VALUE);

  auto start = std::chrono::steady_clock::now();

  try {
    cl::sycl::buffer<float, 1> dots_buf(dots.data(), dots.size());
    mkl::rng::generate(distribution, engine, dots.size(), dots_buf);

    cl::sycl::buffer<int, 1> reached_buf(&reached, 1);

    queue.submit([&](cl::sycl::handler& cgh) {
      auto dots_acc = dots_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto reached_acc = reached_buf.get_access<cl::sycl::access::mode::atomic>(cgh);
      cgh.parallel_for<ComputeIntegral>(cl::sycl::range<1>(count), [=](cl::sycl::id<1> id) {
        float x = dots_acc[3 * id + 0];
        float y = dots_acc[3 * id + 1];
        float z = dots_acc[3 * id + 2];
        if (z <= IntegralFunction(x, y)) {
          reached_acc[0].fetch_add(1);
        }
      });
    });

    queue.wait_and_throw();
  } catch (std::exception e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  float result = static_cast<float>(reached) / count;

  auto end = std::chrono::steady_clock::now();
  float time =
    std::chrono::duration_cast<std::chrono::milliseconds>(
      end - start).count();
  std::cout << "Execution time: " << time << " ms" << std::endl;

  std::cout << "Analytical solution: " <<
    GetIntegralValue(A_VALUE, B_VALUE) << std::endl;
  std::cout << "Calculated soluion: " << result << std::endl;

  return 0;
}